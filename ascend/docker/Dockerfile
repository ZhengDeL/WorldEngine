ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801
FROM ${BASE_IMAGE}

# This is the recipe for the tested layers, not an assertion of bit-identical
# image reconstruction. Keep the prebuilt Ascend/OpenMMLab versions unchanged.
RUN set -eu; \
    /opt/conda/envs/bevformer/bin/python -m pip download --no-deps --only-binary=:all: \
      --dest /tmp/wheels opencv-python-headless==4.11.0.86; \
    wheel=/tmp/wheels/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl; \
    echo "6efabcaa9df731f29e5ea9051776715b1bdd1845d7c9530065c7951d2a2899eb  $wheel" | sha256sum -c -; \
    for env_name in bevformer sparse4d; do \
      python_bin=/opt/conda/envs/$env_name/bin/python; \
      "$python_bin" -m pip uninstall -y opencv-python opencv-python-headless; \
      "$python_bin" -m pip install --no-deps "$wheel"; \
    done; \
    rm -rf /tmp/wheels

RUN /opt/conda/bin/conda create --yes --prefix /opt/worldengine-geo \
      --override-channels --channel conda-forge \
      python=3.10 numpy=1.23.5 rasterio=1.3.11 fiona=1.9.6

COPY extra_requirements.txt /tmp/worldengine_extra_requirements.txt
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/conda/envs/bevformer/bin/python \
      -m pip install --no-deps --no-cache-dir --index-url https://pypi.org/simple \
      --target /opt/worldengine-deps -r /tmp/worldengine_extra_requirements.txt

ENV PYTHONPATH=/opt/worldengine-deps:/opt/worldengine-geo/lib/python3.10/site-packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/worldengine-geo/lib:${LD_LIBRARY_PATH}
