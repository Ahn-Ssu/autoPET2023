#FROM python:3.9-slim
FROM pytorch/pytorch:latest


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

# RUN rm -rf /opt/algorithm

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm/

# ENV PATH="/home/miniconda3/bin:${PATH}"
ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN ls -a
RUN rm -rf /opt/algorithm/*

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm inference.py /opt/algorithm/
COPY --chown=algorithm:algorithm model.py /opt/algorithm/
COPY --chown=algorithm:algorithm last.ckpt /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/


RUN pip install numpy scipy easydict scikit-learn
RUN python -c 'import torch; print(torch.__version__)'
RUN python -m pip --default-timeout=1000 install torch torchvision torchaudio 
# RUN python -m pip --default-timeout=1000 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install -U scikit-learn 
RUN python -m pip install --user -rrequirements.txt
# RUN python -m pip install --user numpy==1.22.0
# RUN python import numpy; print(f'{numpy.__version__=}')
RUN python -c "import numpy; print('numpy version:', numpy.__version__)"



ENTRYPOINT python -m process $0 $@
