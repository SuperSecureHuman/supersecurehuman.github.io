---
title: "Setting up Dockerized Slurm Cluster on Raspberry Pis"
excerpt: "Creating a SLURM cluster on raspberry pi with dockerized worker nodes"
tags:
    - cluster
    - hpc

header:
    overlay_image: "https://i.imgur.com/Ue4NYXH.jpeg"
    overlay_filter: 0.5
---


## Background

Here is the scenario - you need to find a way to share the compute power that 'you' (or your organization) possess, in an effective manner. In my case, we need to share the GPU servers we have with an entire department of researchers. However, the catch is this - you are not allowed to touch the bare metal, you can do whatever you need to, but only within containers.

At present, a group of researchers are allocated one node to work on, which proves to be very inefficient. Top research institutes and 'clustering' devs use SLURM, but by nature, SLURM functions on bare metal... Here is my journey to create a SLURM cluster, using Docker containers on multiple nodes that just works (or rather gets the job done, at least).

## What is SLURM and how does it work

Wikipedia tells - "The Slurm Workload Manager, formerly known as Simple Linux Utility for Resource Management, or simply Slurm, is a free and open-source job scheduler for Linux and Unix-like kernels, used by many of the world's supercomputers and computer clusters."

That's precisely what SLURM does! If you have 10 machines and 100 users need to share them, the users can create a modified version of a bash script to launch their workload, and SLURM takes care of resource allocation, multi-node workloads, and everything else. You can also specify how much GPU, CPU cores, RAM, etc. you need. This ensures that all resources are shared in a fair and efficient manner.

The primary advantage of using SLURM, as opposed to sharing logins, is that it ensures optimal utilization of the computing resources. For example, if you have 10 nodes and each node has been allocated to a group of 10 people, there is a possibility that one node might be over-utilized while another node may have no jobs running on it. SLURM solves this issue by allocating free nodes as jobs come in, queuing jobs if the compute is not yet available, and providing more cool features.

![](https://i.imgur.com/B3AWiOI.jpeg)

## Senario here

I want to try and test SLURM with a local cluster, at least attempting to replicate a production setup. I don't have spare cash lying around to acquire a couple of nodes with shiny new Nvidia GPUs. I mean, I could technically experiment in a production system... Right?

I have a basic Raspberry Pi cluster on which all experiments will be run.

## Hardware in use

* 2x Raspberry Pi 4 (8GB variants)
* SD Cards for boot
* USB stick on each of them - Preferably same size (Optional, but highly recommended)
* A router / Switch with copule of ethernet cables
* Power to everything
* Fan (optional)
* Coffee (Must)

## Setup Raspi

On each of the PIs, install any server operating system you want. Ensure you are able to SSH into all of the PIs you use... You can manually image the PI if you just have 2 or 3.. If you have more, I suggest you to look into some automated way to image all the SD cards you want in parallel.

Furthermore, you can set up Ansible on PIs if you have too many to handle... In my case, since it was only 2, I did not have to work with it. Note that you might want to set up Ansible playbooks anyway, to remember what you did earlier.


![](https://i.imgur.com/IRGnILj.png)

Now, plug in the 32GB USB stick into both nodes. We are going to create a network file system that all the nodes can share files on. In the case of a cluster, it has become a norm to have a common filesystem. This makes everything easy - managing files, quotas, running workloads, etc.

Format both in an identical way, and mount them.

You have the option of partitioning the SD cards, but dedicated storage is much better (besides, these USB sticks are much faster than the SD cards).

In case your network has some issue trying to ping each Raspi with their hostname, consider setting up the hosts file for each Raspi so that it will be much easier later on.

Make sure each node can access each other, and only then go to the next step!

My case - 192.168.0.106 - rpi1
        - 192.168.0.108 - rpi2

## Setup GlusterFS

GlusterFS is a scalable network file system that works on any hardware you throw at it. You can pool drives over the network, create failsafe network storage, and more. In our case, it will act as the primary storage for any common files that our Pis and our SLURM system need to work with.


To install GlusterFS:

```bash
sudo apt-get install -y glusterfs-server glusterfs-client
```

Start the GFS service:

```bash
sudo systemctl start glusterd.service
```

From the master/head node, probe all other nodes in your network, then probe your master from the other nodes:

```bash
sudo gluster peer probe rpi1 # from rpi2
sudo gluster peer probe rpi2 # from rpi1
```


Create a folder inside your mount (USB stick). In my case, the USBs are mounted at /mnt, and there is a folder in them called /mnt/gv0. You can choose any name, but being consistent helps... everywhere

After this, from the master node, create a GlusterFS volume:


```bash
sudo gluster volume create gv0 replica 2 rpi1:/mnt/gv0 rpi2:/mnt/gv0
```

Once the volume named gv0 is created, start it:


```bash
sudo gluster volume start gv0
sudo gluster volume info
```

![GFS info](https://i.imgur.com/1sCxgMi.jpeg)


Now, we need to mount our fresh volume somewhere. I've created a folder /gfsv to keep this as 'gluster file system volume':

Then mount our Gfs volume, on all the nodes

```bash
sudo mkdir /gfsv
sudo mount -t glusterfs rpi1:/gv0 /gfsv/

# Mofify the permissions - based on your envoironment... In my 'lab' no one cares, so...
sudo chmod 777 /gfsv/
```

After a file copy test, you can view in both the nodes that the file appears. As a bonus, while you are copying some large file over, you can also monitor the network activity!

Do note that, doing this over a gigabit connection might be a bottle neck in production senario. Ideally you should be running a 10gig lan, or even better infiniband for the best performance


## Creating the Containers

Make to have docker up and running, there are million guides on it :)

SLRUM, atleast in the barebones state needs 2 components to work - Slurm Master, Slurm worker nodes.

Slurm master node, takes care of resource allocation (and in our case, it also will be our login node through which we will submit jobs)

### Master node docker file

```dockerfile
#Base OS
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
# Install needed packages
RUN apt install munge nano build-essential git mariadb-server wget slurmd slurm-client slurmctld sudo openssh-server -y

# Add a user to manage everything, and be able to ssh (will be handy everywhere)
RUN useradd -m admin -s /usr/bin/bash -d /home/admin && echo "admin:admin" | chpasswd && adduser admin sudo && echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# This step, is more like a hack, I am sure there is a better way to do it. 
RUN echo 'OPTIONS="--force --key-file /etc/munge/munge.key"' > /etc/default/munge


# You could bind mount them also
COPY slurm.conf /etc/slurm/

# Script will be ther below
COPY docker-entrypoint.sh /etc/slurm/

#EXPOSE 6817 6818 6819 3306 

RUN chmod +x /etc/slurm/docker-entrypoint.sh 

ENTRYPOINT ["/etc/slurm/docker-entrypoint.sh"]
```

docker-entrypoint.sh

```bash
# This key is form the bind mount. This will be explained later
sudo chown munge:munge /etc/munge/munge.key
sudo chown munge:munge /etc/munge
sudo service munge start
sudo service slurmctld start
sudo service ssh start

# Just so that the container doesnot stop.. you could just start slurmctld in deamon mode if you want
tail -f /dev/null
```

### Worker node docker file

```dockerfile
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt install munge nano build-essential git wget slurmd slurm-client sudo openssh-server -y

RUN useradd -m admin -s /usr/bin/bash -d /home/admin && echo "admin:admin" | chpasswd && adduser admin sudo && echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN echo 'OPTIONS="--force --key-file /etc/munge/munge.key"' > /etc/default/munge

COPY slurm.conf /etc/slurm/
COPY cgroup.conf /etc/slurm/
COPY docker-entrypoint.sh /etc/slurm/

#EXPOSE 6817 6818 6819  

RUN chmod +x /etc/slurm/docker-entrypoint.sh 

ENTRYPOINT ["/etc/slurm/docker-entrypoint.sh"]
```

docker-entrypoint.sh

```bash
#!/bin/bash

sudo chown munge:munge /etc/munge/munge.key
sudo chown munge:munge /etc/munge
sudo service munge start
sudo slurmd -N $(hostname)
sudo service ssh start



tail -f /dev/null
```

### Other needed files

1. You need a common munge key accross all the nodes. In my case, I generated one with a temperory instance of ubuntu container, coped that over to /gfsv/etc/munge/munge.key. Since I am mouting this volumne in all the containers, the above 'hack' was needed to make it work.

2. You need a SLURM config file.. The usual config remains, you can use the slurm congigurator, but, due to certain restrictions in containers, the cgroup based plugins wont work out of the box. For this experumebt I will be not using them. Later, I will work on getting cgroup working within slurm.

slurm.conf

```conf
ClusterName=asai_cluster
SlurmctldHost=slurm-master
ProctrackType=proctrack/linuxproc
#ProctrackType=proctrack/cgroup
PrologFlags=Contain
ReturnToService=1
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/spool/slurmd
#SlurmUser=slurm
SlurmdUser=root

StateSaveLocation=/var/spool/slurmctld
#TaskPlugin=task/affinity,task/cgroup
TaskPlugin=task/none

NodeName=slurm-worker-[1-2] CPUs=4 Sockets=1 CoresPerSocket=4 ThreadsPerCore=1 State=UNKNOWN
PartitionName=debug Nodes=ALL Default=YES MaxTime=INFINITE State=UP
```

## Docker Networking - MACVLAN

Something to keep in mind here is that using the default Docker way of networking - bridge networking is not a good idea. In the case of multi-node workloads, random ports will be allocated, and there will be complications when trying to auto-map them to the host. Instead, the best way to do this is to make the container behave like a bare-metal node to the network. MACVLAN networking in Docker accomplishes this. The container will now be visible in the network as a dedicated node.

To set up MACVLAN networking in both of your nodes, enter:



```bash
docker network create -d macvlan \  # driver name
--subnet 192.168.0.0/24 \  # subnet of your network
--gateway 192.168.0.1 \  # default gateway of your network
-o parent=eth0 slurm_network  # name for this network
```

Now, check your network settings and preferably exclude some IPs from its DHCP range, or reserve those addresses. These addresses can be used for the containers.

On my network, the DHCP range is from `192.168.0.100` to `192.168.0.200`

I have these IPs in plan

* Slurm Master - 192.168.0.201
* Slurm Worker 1 - 192.168.0.202
* Slurm Worker 2 - 192.168.0.203

## Simple script to build these containers

Now, copy these docker scripts to your shared volume. I have placed them in a shared folder named build_files.

A script named build.sh is used to build these containers and tag them.

```sh
#!/bin/bash

# change to the master directory
cd master
# build the Docker container for the master
docker build -t slurm_master .
# change to the worker directory
cd ../node
# build the Docker container for the worker
docker build -t slurm_worker .
```

### Just a mini setup recap

* Make sure to have the munge key in the right place
* Make sure you have created a macvlan network and have the IPs ready for your use case
* Working shared filesystem
* Built the needed containers on all the hosts (you could push them to a registry, but I choose to build them locally)

## Launching the containers

Launch the containers! First launch the worker nodes, then launch the master node.

```bash
docker run -it -d --name slurm_worker_1 \
--hostname slurm-worker-1 --network macvlan \
--add-host slurm-master:192.168.0.201 \
--add-host slurm-worker-2:192.168.0.203 \
--user admin \
--ip 192.168.0.202 \
-v /gfsv/home:/home/ \
-v /gfsv/munge:/etc/munge/ \
-e SLURM_NODENAME=slurm-worker-1 \
slurm_worker
```

```bash
docker run -it -d --name slurm_worker_2 \
--hostname slurm-worker-2 --network macvlan \
--add-host slurm-master:192.168.0.201 \
--add-host slurm-worker-1:192.168.0.202 \
--user admin --ip 192.168.0.203 \
-v /gfsv/home:/home/ \
-v /gfsv/munge:/etc/munge/ \
-e SLURM_NODENAME=slurm-worker-2 \
slurm_worker
```

```bash
docker run -it -d --name slurm_master \
--hostname slurm-master --network macvlan \
--add-host slurm-worker-1:192.168.0.202 \
--add-host slurm-worker-2:192.168.0.203 \
--user admin --ip 192.168.0.201 \
-v /gfsv/home:/home/ \
-v /gfsv/munge:/etc/munge/ \
slurm_master
```

Our setup is now like

![](https://i.imgur.com/F7lX7I9.png)

### Note on common home folder

Here, the home folder is shared, but since the user was created during the Docker build process, the home folder of 'admin' is lost.

Bash into the container and perform the following steps to restore a functional home folder:

```bash
sudo su
mkdir /home/admin
chown -R admin:admin /home/admin
```

Logout and log back in; you should now have a working home folder. You only need to do this in the master containers, since the UID and GID will be the same.

### Note on creating users

There is still a need to create multiple users; in that case, make sure the UIDs and GIDs are consistent across all the containers. To overcome this, you can use LDAP to manage the UIDs and GIDs, and configure all containers to use LDAP.

### Note on MACVLAN and attempting to SSH into the container

In a MACVLAN scenario, the host cannot contact the container network. All other devices including other containers, except the host, can access the container. This is a security feature. If you require such control for some reason, consider using IPVLAN.

## Testing SLURM base features

First ssh into your master node

![](https://i.imgur.com/TEzUk0L.jpeg)

Type `sinfo`

```term
admin@slurm-master:~$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
debug*       up   infinite      2   idle slurm-worker-[1-2]
admin@slurm-master:~$
```

Type `scontrol show node`

```term
NodeName=slurm-worker-1 Arch=aarch64 CoresPerSocket=4
   CPUAlloc=0 CPUTot=4 CPULoad=0.00
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=(null)
   NodeAddr=slurm-worker-1 NodeHostName=slurm-worker-1 Version=21.08.5
   OS=Linux 6.8.0-1004-raspi #4-Ubuntu SMP PREEMPT_DYNAMIC Sat Apr 20 02:29:55 UTC 2024
   RealMemory=1 AllocMem=0 FreeMem=461 Sockets=1 Boards=1
   State=IDLE ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=debug
   BootTime=2024-05-28T11:28:13 SlurmdStartTime=2024-05-30T13:11:37
   LastBusyTime=2024-05-30T13:21:06
   CfgTRES=cpu=4,mem=1M,billing=4
   AllocTRES=
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s

NodeName=slurm-worker-2 Arch=aarch64 CoresPerSocket=4
   CPUAlloc=0 CPUTot=4 CPULoad=0.00
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=(null)
   NodeAddr=slurm-worker-2 NodeHostName=slurm-worker-2 Version=21.08.5
   OS=Linux 6.8.0-1004-raspi #4-Ubuntu SMP PREEMPT_DYNAMIC Sat Apr 20 02:29:55 UTC 2024
   RealMemory=1 AllocMem=0 FreeMem=456 Sockets=1 Boards=1
   State=IDLE ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=debug
   BootTime=2024-05-28T11:28:17 SlurmdStartTime=2024-05-30T13:11:01
   LastBusyTime=2024-05-30T13:21:06
   CfgTRES=cpu=4,mem=1M,billing=4
   AllocTRES=
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
```

Now, to test whether our cluster is actually working, we can submit a very simple job on 2 nodes.

This command will execute the specified user script on 2 nodes, and display the output in the terminal.

The flag -N2 indicates the number of nodes on which to run this command. In my case, I have 2.

![](https://i.imgur.com/xINlPMs.jpeg)


We can see that slurm has run this command on both of our nodes and returned the hostname to us

## Installing dependencies the SLURM way

In order to run anything, we would need to install dependencies. In a normal scenario, we would need to use "apt install" to install the packages we need. However, when we have SLURM, we can use "srun" to execute the same install command on all nodes, and the dependencies will be installed on all nodes in parallel.

We are going to test an MPI program.

```bash
srun -N2 sudo apt install python3 python3-pip python3-mpi4py python3-numpy libopenmpi-dev -y
```

Once done, we can submit a workload!

## Creating an MPI program in Python

Here is a simple Python script that uses MPI to calculate the value of Pi.

```python
from mpi4py import MPI
from math   import pi as PI
from numpy  import array

def comp_pi(n, myrank=0, nprocs=1):
    h = 1.0 / n
    s = 0.0
    for i in range(myrank + 1, n + 1, nprocs):
        x = h * (i - 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h

def prn_pi(pi, PI):
    message = "pi is approximately %.16f, error is %.16f"
    print  (message % (pi, abs(pi - PI)))

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

n    = array(0, dtype=int)
pi   = array(0, dtype=float)
mypi = array(0, dtype=float)

if myrank == 0:
    _n = 10000000 # Enter the number of intervals - 
    n.fill(_n)
comm.Bcast([n, MPI.INT], root=0)
_mypi = comp_pi(n, myrank, nprocs)
mypi.fill(_mypi)
comm.Reduce([mypi, MPI.DOUBLE], [pi, MPI.DOUBLE],
            op=MPI.SUM, root=0)
if myrank == 0:
    prn_pi(pi, PI)
```

You could modify the number of intervals to use up more compute and get more precise value.

## Creating a SLURM submit script

Creating a perfect SLURM script, depends fully on the env, Usually your cluster docs will have the best guide for it. In this case, all we need is

```bash
#!/bin/bash
#SBATCH --ntasks=8
#SBATCH -N2
cd $SLURM_SUBMIT_DIR
mpiexec -n 8 python3 calc.py
```

Here, we tell slurm that we need to execute a total of 8 processes, over 2 nodes. SLURM will take care of everything else.

To submit it, type `sbatch script.sh`

## Launching the SLURM script, Obeserving the nodes for usage

![](https://i.imgur.com/msKKI2Y.jpeg)

## Conslusion

We now have a "working" slurm cluster. It satisfies the constraints I had, and gives a very nice way of sharing compute. Given a constrained / protected network, this setup is completely viable to run!

I now have a Pi cluster, that can calculate the value of PI.

Here is how my final master piece looks (pls ignore the industrial grade cooling equipment, also my home lab my rules ;) )

![](https://i.imgur.com/Ue4NYXH.jpeg)

## Next Steps...

* Get SlurmDB running
* Compile SLURM, rather than the package manager way. Use the latest version
* Get Cgroups sorted out
* Setup LDAP auth
* Setup slurm-web