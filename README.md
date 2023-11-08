# 목표
video를 보고, mujoco환경에서의 skill(action space)에 해당하는 정보를 추출하고 싶다.
이를 위해 EmbodiedPose를 재현해야 하는데, 해당 코드는 추가로 2개의 repo 재현을 요구한다.

* UHC: Universal Humanoid Controller (https://github.com/ZhengyiLuo/UniversalHumanoidControl)
* HybrIK (https://github.com/Jeff-sjtu/HybrIK)
* EmbodiedPose (https://github.com/zhengyiluo/EmbodiedPose)

<br/><br/>


# Universal Humanoid Control
Mujoco 환경에서 기존의 humanoid 모델 대신 SMPL 모델을 사용하는 환경

## Installation
### 기본 사항
Conda 환경에서 실행했다. 기본적으로 Headless환경이긴한데, 혹시나 몰라서 Screen이 존재하는 서버에서 재현했다.
### 환경세팅
* Ubuntu 20.04
* nvidia driver 535.104.05
* CUDA toolkit 11.8
* x11forwarding 가능해야함.


### install app
```
sudo apt install libglew-dev
sudo apt-get install python3-opencv
sudo apt install ffmpeg
```
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

### Environment Variable
mujoco를 위한 환경변수 설정이다. 새로운 터미널 창을 열 때마다 입력해도 되고, 아니면 ~/.baserc 파일에 직접 적어도 된다.

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.535.104.05
```


## Troubleshooting
### cython error
cython 버전 호환문제. 아래 코드 실행시키면 됨.
```
pip install "cython<3"
```

### mujooco errors
```
opengl 1.5 or higher
nvidia driver doesn't match. need ENV path to direct to nvidia driver
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.535.104.05
```
<br/><br/>



# EmbodiedPose
EmbodiedPose를 실행하기 위해서는 먼저 HybrIK를 설치해야하는데, 여기서부터 호환성 문제가 생길 수 있다.

## 추가 설치 ( VPoser )
`data/vposer/vposer_v1_0`이라는 것이 필요한데, 어디서 다운받는지 repo에는 설명이 되어 있지 않다. 찾다보니 SMPLIFY (https://github.com/vchoutas/smplify-x)에서 Dowload (https://smpl-x.is.tue.mpg.de/download.php) 할 수 있다.


## numpy에서 bool 문제 (chumpy)
chumpy라는 package에서 numpy에서 bool을 가져오면서 발생하는 에러.
근데 이게 원래 numpy 1.23.1은 됬다고 하는데 현재 안되는 듯. 그래서 numpy 1.22로 다운그레이드 하면 된다.
```
pip install numpy==1.22
```


## Troubleshooting (HybrIK)
### pytorch3d 문제
대략 이런 에러가 뜨는데 이건 pytorch3d를 pip install pytorch3d 로 설치하면 구버전이 설치되면서 발생하는 문제.
/lib/python3.8/site-packages/pytorch3d/_C.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZNK2at6Tensor7is_cudaEv
pytorch3d의 github에서 최신 stable 버전을 설치하면 해결됨

```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

혹은 0.7.3 버전을 설치하면 된다.
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.3"
```


### pytorch3d 설치시 setup.py에서 hang 걸리는 경우
이유는 모르겠지만, 설치 도중 무한로딩이 걸리면 아래 환경변수를 추가하면 해결이 된다.
```
export PYTORCH_NO_NINJA=1
```
특이하게도 해당 환경변수는 pytorch3d 설치할 때만 선언해주면 되는 것 같다.

### 'nvidia-smi' vs 'nvcc --version'
nvidia-smi에는 GPU driver가 nvcc에는 cuda 버전이 보이는데, 만약 pytorch3d 설치시 문제가 발생하면 아마 nvcc --version에 보이는 cuda 버전이 맞지 않아서 일 확률이 놉다.

* nvidia-smi: driver API version
* nvcc --version: runtime API version


혹시나 CUDA를 최신버전 혹은 특정 버전으로 깔아도 초기에 설치한 버전이 나타날 경우, PATH를 추가해주면 된다.
```
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
```

```
sudo sh -c 'echo "export PATH=$PATH:/usr/local/cuda-11.8/bin">> /etc/profile'
sudo sh -c 'echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64">> /etc/profile'
sudo sh -c 'echo "export CUDARDIR=/usr/local/cuda-11.8">> /etc/profile'
```


### pretrained_models 에 모델 파일명이 존재하지 않는 것.
README에는 분명 'hybrik_hrnet.pth'를 넣으라고 되어있지만, Github의 README에 존재하는 pretrained model file들 중에서는 동일한 이름을 가진 파일이 없다. 대신 issue에 보면 
(https://github.com/Jeff-sjtu/HybrIK/issues/185)

hybrik_hrnet48_w3dpw.pth 를 hybrik_hrnet.pth로 파일명만 바꿔서 사용하면 일단 error는 발생하지 않는다고 한다. 그러나 원작자가 해당 방식이 올바른 것인지 알려주지 않았기 때문에 유의해야 한다.


### CV2 문제
아래의 error는 openCV 관련 에러다. openCV는 pip 가 아니라 apt-get으로 설치해야한다.
```
Could not find encoder for codec id 27: Encoder not found
```

설치 코드
```
sudo apt-get install python3-opencv
```


## ffmpeg
```sh: 1: ffmpeg: not found``` 라는 error 메시지와 동시에, Model의 output에 아무것도 없는 경우가 뜰 때가 있다. 
<br/> 이는 ffmpeg라는 별도의 apt를 깔아줘야하기 때문이다.
```
sudo apt install ffmpeg
```





# Troubleshooting while running codes
## Render 및 Viewer 사용시
`viewer.render()`이든 `sim.render()`이든 rendering해서 image를 저장할 때, 아래 error가 나올 때가 있다.
```
Found 3 GPUs for rendering. Using device 0.
Could not make EGL context current
Traceback (most recent call last):
  File "testing_envs/test_uhc.py", line 216, in <module>
    running_env()
  File "testing_envs/test_uhc.py", line 85, in running_env
    print(f"sim.render : {sim.render(width=width, height=height)}")
  File "mjsim.pyx", line 156, in mujoco_py.cymj.MjSim.render
  File "mjsim.pyx", line 158, in mujoco_py.cymj.MjSim.render
  File "mjrendercontext.pyx", line 46, in mujoco_py.cymj.MjRenderContext.__init__
  File "mjrendercontext.pyx", line 114, in mujoco_py.cymj.MjRenderContext._setup_opengl_context
  File "opengl_context.pyx", line 130, in mujoco_py.cymj.OffscreenOpenGLContext.__init__
RuntimeError: Failed to initialize OpenGL

```
`MjViewer(sim)`을 사용해서 viewer를 띄우고 싶다면 아래 환경변수를 추가해야한다. 근데 웃긴건 이렇게 사용하면 이후에 Viewer를 사용하지 않고 `sim.render()`를 바로 사용한다면 `unset LD_PRELOAD`해줘야함.
```
unset LD_PRELOAD
```

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.535.104.05
```

