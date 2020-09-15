Bootstrap: library
From: ubuntu:18.04

%setup
    mkdir ${SINGULARITY_ROOTFS}/opt/DeepWrist
    cp -R classifier ${SINGULARITY_ROOTFS}/opt/DeepWrist
    cp -R localizer ${SINGULARITY_ROOTFS}/opt/DeepWrist
    cp -R utils ${SINGULARITY_ROOTFS}/opt/DeepWrist
    cp requirements.txt ${SINGULARITY_ROOTFS}/opt/DeepWrist/


%post
    apt-get -y update
    apt-get install -y software-properties-common
    add-apt-repository universe
    apt-get -y update
    apt-get -y install wget
    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
    rm Miniconda3-latest-Linux-x86_64.sh
    export PATH=/opt/conda/bin:${PATH}
    apt-get install -y git
    apt-get install -y python3-pip
    pip install -r /opt/DeepWrist/requirements.txt

