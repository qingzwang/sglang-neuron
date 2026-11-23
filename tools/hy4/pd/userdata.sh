#!/bin/bash
# instance-store NVMe as RAID0 at /mnt/nvme (as on the dev node)
DEVS=$(lsblk -d -n -o NAME,MODEL | awk '/Instance Storage/ {print "/dev/"$1}')
N=$(echo $DEVS | wc -w)
mdadm --create /dev/md0 --level=0 --raid-devices=$N $DEVS --force --run
mkfs.ext4 -F -E nodiscard,lazy_itable_init=1 /dev/md0
mkdir -p /mnt/nvme && mount /dev/md0 /mnt/nvme && chmod 1777 /mnt/nvme
id ssm-user >/dev/null 2>&1 || useradd -m -u 1001 -s /bin/bash ssm-user
echo "ssm-user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ssm-agent-users
touch /mnt/nvme/.ready
