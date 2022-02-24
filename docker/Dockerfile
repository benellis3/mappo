FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
MAINTAINER Christian Schroeder de Witt

RUN pip install numpy scipy pyyaml matplotlib
RUN pip install imageio pygame
RUN pip install tensorboard-logger
RUN pip install ruamel.base ryd

RUN mkdir /install
WORKDIR /install

RUN pip install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python setup.py install

# Install pymongo
RUN pip install pymongo

## -- SMAC
ENV smac_ver 1
RUN pip install git+https://github.com/benellis3/smac.git@stochastic_test
ENV SC2PATH /home/minsun/pymarl/3rdparty/StarCraftII
RUN pip install wandb
RUN git clone https://github.com/koulanurag/ma-gym.git /install/ma-gym && cd /install/ma-gym && pip install -e .

ARG UID
RUN useradd -u $UID --create-home minsun
USER minsun
WORKDIR /home/minsun/pymarl

RUN echo 'set editing-mode vi' >> /home/minsun/.inputrc
RUN echo 'set keymap vi' >> /home/minsun/.inputrc
