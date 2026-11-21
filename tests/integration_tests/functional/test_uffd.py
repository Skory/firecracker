# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test UFFD related functionality when resuming from snapshot."""

import os
import re
import uuid
from pathlib import Path

import pytest
import requests

from framework import defs
from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel
from framework.utils import Timeout, check_output
from framework.vm_backend import VM_BACKEND_KVM, pin_vm_backend

# KVM-only: snapshotting is not available on enclave VMs.
pytestmark = pin_vm_backend(VM_BACKEND_KVM)


@pytest.fixture(scope="function", name="snapshot")
def snapshot_fxt(microvm_factory, vm_backend, guest_kernel, rootfs):
    """Create a snapshot of a microVM."""

    basevm = microvm_factory.build(guest_kernel, rootfs, backend=vm_backend)
    basevm.spawn()
    basevm.basic_config(vcpu_count=2, mem_size_mib=256)
    basevm.add_net_iface()

    # Add a memory balloon.
    basevm.api.balloon.put(
        amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=0
    )

    basevm.start()

    # Create base snapshot.
    snapshot = basevm.snapshot_full()
    basevm.kill()

    yield snapshot


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_bad_socket_path(uvm, snapshot):
    """
    Test error scenario when socket path does not exist.
    """
    vm = uvm
    vm.spawn()
    jailed_vmstate = vm.create_jailed_resource(snapshot.vmstate)

    expected_msg = re.escape(
        "Load snapshot error: Failed to restore from snapshot: Failed to load guest "
        "memory: Error creating guest memory from uffd: Failed to connect to UDS Unix stream: No "
        "such file or directory (os error 2)"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        vm.api.snapshot_load.put(
            mem_backend={"backend_type": "Uffd", "backend_path": "inexistent"},
            snapshot_path=jailed_vmstate,
        )

    vm.mark_killed()


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_unbinded_socket(uvm, snapshot):
    """
    Test error scenario when PF handler has not yet called bind on socket.
    """
    vm = uvm
    vm.spawn()

    jailed_vmstate = vm.create_jailed_resource(snapshot.vmstate)
    socket_path = os.path.join(vm.path, "firecracker-uffd.sock")
    check_output("touch {}".format(socket_path))
    jailed_sock_path = vm.create_jailed_resource(socket_path)

    expected_msg = re.escape(
        "Load snapshot error: Failed to restore from snapshot: Failed to load guest "
        "memory: Error creating guest memory from uffd: Failed to connect to UDS Unix stream: "
        "Connection refused (os error 111)"
    )
    with pytest.raises(RuntimeError, match=expected_msg):
        vm.api.snapshot_load.put(
            mem_backend={"backend_type": "Uffd", "backend_path": jailed_sock_path},
            snapshot_path=jailed_vmstate,
        )

    vm.mark_killed()


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_valid_handler(uvm, snapshot):
    """
    Test valid uffd handler scenario.
    """
    vm = uvm
    vm.memory_monitor = None
    vm.spawn()
    vm.restore_from_snapshot(snapshot, resume=True, uffd_handler_name="on_demand")

    # Inflate balloon.
    vm.api.balloon.patch(amount_mib=200)

    # Verify if the restored guest works.
    vm.ssh.check_output("true")

    # Deflate balloon.
    vm.api.balloon.patch(amount_mib=0)

    # Verify if the restored guest works.
    vm.ssh.check_output("true")


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_malicious_handler(uvm, snapshot):
    """
    Test malicious uffd handler scenario.

    The page fault handler panics when receiving a page fault,
    so no events are handled and snapshot memory regions cannot be
    loaded into memory. In this case, Firecracker is designed to freeze,
    instead of silently switching to having the kernel handle page
    faults, so that it becomes obvious that something went wrong.
    """

    vm = uvm
    vm.memory_monitor = None
    vm.spawn()

    # We expect Firecracker to freeze while resuming from a snapshot
    # due to the malicious handler's unavailability.
    try:
        with Timeout(seconds=30):
            vm.restore_from_snapshot(
                snapshot, resume=True, uffd_handler_name="malicious"
            )
            assert False, "Firecracker should freeze"
    except (TimeoutError, requests.exceptions.ReadTimeout):
        vm.uffd_handler.mark_killed()


# Written into guest tmpfs before the snapshot, so it lives in guest memory
# pages that every microVM restored from the snapshot starts out sharing.
SHARED_PAGE_FILE = "/dev/shm/shared-page"
SHARED_PAGE_DATA = "written-before-snapshot"
PAGE_SIZE = 4096


@pytest.fixture(name="shared_file")
def shared_file_fxt():
    """A shared memory file on the same tmpfs as the jails."""
    path = Path(defs.DEFAULT_TEST_SESSION_ROOT_PATH) / f"shared-{uuid.uuid4().hex}.mem"
    path.touch()
    yield path
    path.unlink()


def restore_shared(microvm_factory, snapshot, shared_file):
    """Restore a microVM from `snapshot` in uffd shared memory mode."""
    vm = microvm_factory.build()
    vm.memory_monitor = None
    vm.spawn()
    vm.restore_from_snapshot(
        snapshot,
        resume=True,
        uffd_handler_name="on_demand",
        uffd_shared_file=shared_file,
    )
    return vm


def assert_shared_file_unchanged(shared_file, mem_file):
    """Check that every page of the shared file is either still a hole or
    holds the snapshot's bytes, and return how many pages are filled."""
    filled = 0
    zero_page = bytes(PAGE_SIZE)
    with open(shared_file, "rb") as shared, open(mem_file, "rb") as mem:
        offset = 0
        while page := shared.read(PAGE_SIZE):
            expected = mem.read(PAGE_SIZE)
            if page != zero_page:
                assert page == expected, f"shared file changed at offset {offset:#x}"
                filled += 1
            offset += PAGE_SIZE
    return filled


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
def test_shared_memory_mode(microvm_factory, guest_kernel, rootfs, shared_file):
    """
    Test two microVMs restored from one snapshot in uffd shared memory mode.

    Both boot, guest memory stays private to each microVM, and guest writes
    never reach the shared file.
    """
    basevm = microvm_factory.build(guest_kernel, rootfs)
    basevm.spawn()
    basevm.basic_config(vcpu_count=2, mem_size_mib=256)
    basevm.add_net_iface()
    basevm.start()
    basevm.ssh.check_output(f"echo {SHARED_PAGE_DATA} > {SHARED_PAGE_FILE}")
    snapshot = basevm.snapshot_full()
    basevm.kill()

    vm0 = restore_shared(microvm_factory, snapshot, shared_file)
    vm0.ssh.check_output("true")
    filled_after_vm0 = assert_shared_file_unchanged(shared_file, snapshot.mem)
    assert filled_after_vm0 > 0

    vm1 = restore_shared(microvm_factory, snapshot, shared_file)
    vm1.ssh.check_output("true")

    # Both microVMs map the page from the shared file. vm0 overwrites it in
    # place, which must copy it privately, so vm1 still reads the old data.
    for vm in (vm0, vm1):
        _, stdout, _ = vm.ssh.check_output(f"cat {SHARED_PAGE_FILE}")
        assert stdout.strip() == SHARED_PAGE_DATA
    vm0.ssh.check_output(
        f"printf written-by-vm0 | dd of={SHARED_PAGE_FILE} conv=notrunc"
    )
    _, stdout, _ = vm0.ssh.check_output(f"cat {SHARED_PAGE_FILE}")
    assert stdout.startswith("written-by-vm0")
    _, stdout, _ = vm1.ssh.check_output(f"cat {SHARED_PAGE_FILE}")
    assert stdout.strip() == SHARED_PAGE_DATA

    assert_shared_file_unchanged(shared_file, snapshot.mem)
