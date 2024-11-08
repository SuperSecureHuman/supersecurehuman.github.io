---
title: "Working with GlusterFS and NFS (NFS-Ganesha)"
excerpt: "Setting up and working with NFS Ganesha on GlusterFS. This guide will be using raspberry pi's as the nodes, but it should be applicable to any system."
tags:
    - cluster
    - nfs
    - glusterfs
    - kubernetes
    - pvc
    

header:
    overlay_image: ""
    overlay_filter: 0.5
---


Trying to run any sort of cluster workloads needs a shared storage solution. GlusterFS is my prefered setup, works well when you deal with the files directly. When it comes to working with PVC on Kubernetes, it's a bit of a pain. NFS is a better solution for that. NFS-Ganesha is a user-space NFS server that supports NFSv3, NFSv4, and pNFS. It's a good solution for Kubernetes PVCs. This guide will be using raspberry pi's as the nodes, but it should be applicable to any system.

Note that this article is not a argument on which is better, GlusterFS or NFS. This is a learning journey to have a file storage solution that works well with Kubernetes PVCs. In real world scenarios, there may be better solutions, but the principles should be the same.  

## 1. **Setting up GlusterFS**

### Install GlusterFS on All Nodes
```bash
sudo apt-get install -y glusterfs-server glusterfs-client
sudo systemctl start glusterd.service
```

### Add Nodes to the Cluster (on master node)
```bash
sudo gluster peer probe <node-ip>  # Repeat for each node
```

### Create and Start GlusterFS Volume (on master node)
```bash
sudo gluster volume create gv0 replica 3 rpi1:/mnt/gv0 rpi2:/mnt/gv0 rpi3:/mnt/gv0
sudo gluster volume start gv0
sudo gluster volume info
```

---

## 2. **Setting up NFS-Ganesha**

### Install NFS-Ganesha
```bash
sudo apt -y install nfs-ganesha-gluster
sudo mv /etc/ganesha/ganesha.conf /etc/ganesha/ganesha.conf.org  # Backup original
```

### Configure `/etc/ganesha/ganesha.conf`
Edit and replace with the following configuration:
```bash
NFS_CORE_PARAM {
    mount_path_pseudo = true;
    Protocols = 3, 4;
}

EXPORT_DEFAULTS {
    Access_Type = RW;
}

LOG {
    Default_Log_Level = WARN;
}

EXPORT {
    Export_Id = 1;
    Path = "/shared_vol";

    FSAL {
        name = GLUSTER;
        hostname = "192.168.0.105";
        volume = "gv";
    }

    Access_type = RW;
    Squash = No_root_squash;
    Disable_ACL = TRUE;
    Pseudo = "/gv";
    Protocols = 3, 4;
    Transports = "UDP", "TCP";
    SecType = "sys";
}
```

### Start and Enable NFS-Ganesha
```bash
sudo systemctl start nfs-ganesha
sudo systemctl enable nfs-ganesha
```

### Verify NFS Export
```bash
sudo showmount -e <node-ip>  # Expect "RPC: Program not registered" output
```

---

## 3. **Mounting NFS on All Nodes**

### Install NFS Client
```bash
sudo apt -y install nfs-common
```

### Mount NFS Share
```bash
sudo mount -t nfs4 192.168.0.105:/gv /mnt/nfs
```

### Test NFS Mount
Create a file on one node and verify on others:
```bash
touch /mnt/nfs/test.txt
ls /mnt/nfs
```

---

## 4. **Setting up NFS-Ganesha with K3s**

### Unmount NFS
```bash
sudo umount /mnt/nfs
```

### Install NFS CSI Driver for Kubernetes
Follow the [official guide](https://github.com/kubernetes-csi/csi-driver-nfs/blob/18fdc4a39eb451c7f361effb79cd6a7dc5b4d601/docs/install-nfs-csi-driver.md) or run:
```bash
curl -skSL https://raw.githubusercontent.com/kubernetes-csi/csi-driver-nfs/v4.9.0/deploy/install-driver.sh | bash -s v4.9.0 --
kubectl -n kube-system get pod -o wide -l app=csi-nfs-controller
kubectl -n kube-system get pod -o wide -l app=csi-nfs-node
```

### Create Storage Class for NFS
```yaml
# k3s-nfs-pvc.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-csi
provisioner: nfs.csi.k8s.io
parameters:
  server: 192.168.0.105
  share: /gv
reclaimPolicy: Delete
volumeBindingMode: Immediate
```
Apply it:
```bash
kubectl apply -f k3s-nfs-pvc.yaml
kubectl get storageclasses
```

---

## 5. **Testing PVC with NFS**

Create a PVC and Deployment in Kubernetes:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-deployment-nfs
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nfs-csi
  resources:
    requests:
      storage: 100Mi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-nfs
spec:
  replicas: 1
  selector:
    matchLabels:
      name: deployment-nfs
  template:
    metadata:
      labels:
        name: deployment-nfs
    spec:
      containers:
        - name: deployment-nfs
          image: nginx:latest
          volumeMounts:
            - name: nfs
              mountPath: "/mnt/nfs"
      volumes:
        - name: nfs
          persistentVolumeClaim:
            claimName: pvc-deployment-nfs
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

---

## Conclusion

I hope this made your setup of a GlusterFS and NFS-Ganesha cluster easier. This setup should work well with Kubernetes PVCs. If you have any questions or suggestions, feel free to reach out to me. 

