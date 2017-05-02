---
layout: page
permalink: /How to Build your own Deep Learning Machine/
---

## The beginning

After I start my journey on machine learning and deep learning from January this year, finally I've decided that it's time to build my own machine.

My Macbook-Pro is still running good for general machine learning model with small dataset. But when it comes to deep learning and neural network with large dataset, it takes several hours, or days, to train a model.

I used to built my own PC before, so I kinda know what to do to make one. But to build a machine for deep learning, it requires a good GPU. So, I did some research to find the right ingredient for my machine. And here is my conclusion.

![alt text](/pic/pc_parts.jpg "parts for my machine")

### List of my parts
- Motherboard: ASUS PRIME Z270-A
- CPU: Intel CPU Core i5-7500 3.4GHz
- GPU: ASUS NVIDIA GeForce GTX1080Ti
- Memory: CORSAIR DDR4 VENGEANCE LPX Series 16GB×2
- Storage: SanDisk SSD UltraII 480GB 2.5

I also choose **Corsair Carbite 100R Silent ATX** for PC case and **Corsair RM650x 80PLUS GOLD** for power supply to keep the noise as low as possible.

![alt text](/pic/my-pc1.jpg "my machine")

It took me 4-5 hours to assembling all the hardware. And if you notice, the GPU power cord was not connected. (this caused my some trouble later on)

### Time to install software
I want to dedicate this machine only for my study, so I installed **Ubuntu 16.04 LTS** (desktop). And I also need to install **CUDA** and **cuDNN** from NVIDIA to work with **TensorFlow** and other deep learning frameworks. Here is the link of each one.
- [Ubuntu 16.04.2 LTS](https://www.ubuntu.com/download/desktop)
- [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 5.1](https://developer.nvidia.com/cudnn)
- [Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)

I'll skip the OS installation, as it's quite straightforward. You can simply follow this [guide](https://www.ubuntu.com/download/desktop/install-ubuntu-desktop). Just need to make sure that you plug your GPU power supply when you install your OS. And you also need internet access to install other packages.

### Preparation Process
After finish OS installation, you will need to verify whether your OS see your GPU. You can run the `lspci` command in your terminal.
```bash
$ lspci | grep -i nvidia
```
If the message show unknown hardware, run the command `sudo update-pciids` once, and then `lspci | grep -i nvidia` again. This time you should see your GPU. Mine is like this.

```bash
$ lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GP102 HDMI Audio Controller (rev a1)
```

You also need to confirm that you have `gcc` installed in your PC.
```bash
$ gcc --version
```
If you see error, that means you need to install `gcc` again.

Next you will need to install kernel headers and development packages. (This might be done during OS installation, but you can run it again.)
```bash
$ sudo apt-get install linux-headers-$(uname -r)
```
Now we're ready for the real show.

### Installing CUDA Toolkit 8.0
This might be the most tricky part in my installation. NVIDIA do have document for installation guide, but I can tell you that it caused me very confused. [Here is the guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

#### Download CUDA Runfile

Anyway, let start with downloading the installation file from this [URL](https://developer.nvidia.com/cuda-downloads). And I recommend to use **Runfile** for installation, as it doesn't change your linux file path.

![alt text](/pic/cuda-download.jpeg)

And it's better to checksum your file before start installing. [Installer checksums](http://developer2.download.nvidia.com/compute/cuda/8.0/secure/Prod2/docs/sidebar/md5sum.txt?287DFOlRaexHneDsTZKFZCpo_cssBz2b67JmMcU90c1RgMxUxUZ3dqPkmWALYYe2ZFd-xwynJqwXGdebrGbIy5IWZZzo2NIu4FcSdzpVDGYCQ9Fa9ExxAuAfcPgIoW8ScITwdLthjZ9kv0qip2R7DI6niUo)
```bash
$ md5sum <file name>
```

#### Disabling nouveau
Then you have to create file `/etc/modprobe.d/blacklist-nouveau.conf` and add two lines below in the file.
```bash
blacklist nouveau
options nouveau modeset=0
```

Regenerate the kernel initramfs:
```bash
$ sudo update-initramfs -u
```

#### Install CUDA in terminal session
Next step you have to logout from your GUI and enter to terminal session with `ctrl` + `alt` + `F2`. This will switch to terminal TTY2.
```bash
sudo service lightdm stop
sudo sh cuda_<version>_linux.run --override
```

You must `accept` term, but **Don't** install NVIDIA Accelerated Graphics Driver, because it's old version. (We will install new version later)

```bash
.#Term and condition...
.
.
-------------------------------------------
Do you accept the previously read EULA?
accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 375.26?
(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
 [ default is /usr/local/cuda-8.0 ]:

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y

Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: y

Enter CUDA Samples Location
 [ default is /home/username ]:

Installing the CUDA Toolkit in /usr/local/cuda-8.0 ...
```
Don't worry if you did install the Graphics Driver, because we can purge it and upgrade to newer one.

Once the installation finish, you will see the summary report. Now we have to download and install new Graphics Driver from NVIDIA.
In this case, I recommend not to install from Runfile. Here is how to install. (In my case, GTX1080Ti requires version 378 or newer, or my PC will not recognize the GPU)

```bash
$ sudo apt-get purge nvidia-* #first we have to uninstall old driver if any
$ sudo apt-get install nvidia-378 #you can check version number from NVIDIA before install the driver. In my case version 378 support my GPU.
$ sudo service lightdm start # this command will return you back to GUI mode.
```

You will need to add your `PATH` variable and `LD_LIBRARY_PATH`. In my case, I edit file `~/.bashrc` in my home folder and appended following lines.
```bash
#CUDA_PATH
 export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
 export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                        ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Now you can `reboot` your PC.

#### Verification

To verify your installation, you can run `nvcc --version` and if you see message below, that means CUDA is installed.
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```

`nvidia-smi` command will show you if your driver is good for your GPU. Usually, it should display the model number, but in my case it shows only `Graphics Device`. I think it might because of driver does not fully support, but it's okay because I can see some processes running on this GPU. So, it's working. (Actually, I failed to install new driver because I used Runfile, and it ruined my xorg.conf, and the older version couldn't recognize my GPU. Only show `Err` at GPU Name, until I successfully upgrade new driver with `apt-get install nvidia-378`)
```bash
$ nvidia-smi
Tue May  2 23:18:11 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 378.13                 Driver Version: 378.13                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Graphics Device     Off  | 0000:01:00.0     Off |                  N/A |
| 23%   38C    P8    17W / 250W |  10783MiB / 11172MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0       980    G   /usr/lib/xorg/Xorg                             112MiB |
|
+-----------------------------------------------------------------------------+
```

You can also run `nvidia-smi -a` to see more detail of your GPU.

#### CUDA Samples and deviceQuery
Now it's time to test your CUDA Toolkit with Samples file.

```bash
$ cd ~/NVIDIA_CUDA-8.0_Samples
$ make #this will take time to make samples file and store in ~/NVIDIA_CUDA-8.0_Samples/bin
$
~/NVIDIA_CUDA-8.0_Samples$ cd 1_Utilities/deviceQuery
~/NVIDIA_CUDA-8.0_Samples/1_Utilities/deviceQuery$ ./deviceQuery
# This command will show you which GPU that CUDA is running. You should see something like this.
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Graphics Device"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 11172 MBytes (11715084288 bytes)
  (28) Multiprocessors, (128) CUDA Cores/MP:     3584 CUDA Cores
  GPU Max Clock rate:                            1582 MHz (1.58 GHz)
  Memory Clock rate:                             5505 Mhz
  Memory Bus Width:                              352-bit
  L2 Cache Size:                                 2883584 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = Graphics Device
Result = PASS
```
If you get `Result = PASS` then CUDA Toolkit installation is done.

### Install cuDNN
You will need cuDNN to run your neural network with TensorFlow or other Deep learning frameworks. NVIDIA just released cuDNN 6.0, but as TensorFlow support cuDNN5.1, so I decided to go with this version first.

To download the installation file, you need to create NVIDIA Developer account and accept term to download from this [link](https://developer.nvidia.com/cudnn).

Then select `Download cuDNN 5.1 (Jan 20, 2017), for CUDA 8.0` and download `cuDNN v5.1 Library for Linux` and extract file to any folder.

```bash
$ cd <extract folder> # go to the cuda folder that you extract file.
# copy these folders to /usr/local/cuda-8.0
$ sudo cp include/cudnn.h /usr/local/cuda-8.0/include/
$ sudo cp lib64/libcudnn* /usr/local/cuda-8.0/lib64/
```

Done!. Install cuDNN is very simple, but we have no way to verify. We will know shortly when we install TensorFlow.

### Install TensorFlow with GPU
Before install TensorFlow, you will need to install libcupti-dev library with command below.
```bash
$ sudo apt-get install libcupti-dev
```

Then you have to decide how you like to install TensorFlow. You can install in your `native OS`, `virtualenv`, `Docker`, or `Anaconda`.
This link you will show you how to install.
https://www.tensorflow.org/install/install_linux

In my case, I choose `virtualenv`. You can follow the instruction in the link above. And you just need to run `pip3 install --upgrade tensorflow-gpu` to install TensorFlow with GPU.

### Conclusion
I spent my whole weekend to assembling hardware and install software. There're many mistakes I made and I had to reinstall Ubuntu 4-5 times, until I learned to fix each error. Most of my error caused by the Runfile of Graphics Driver conflict with Xorg.conf file.

The document guide from NVIDIA had too many unnecessary steps that sometimes we can skip and should do other way around. There're some good advices on Stackoverflow and other blog explain how to fix each error, which I'm really appreciate. But I found very few of completed How-to for fresh install. So, I think I should share my experience here.

Now I'm ready to challenge more deep learning models, and hope to learn faster with my new machine. I wish I could find somethings I can share with people along my study journey.
