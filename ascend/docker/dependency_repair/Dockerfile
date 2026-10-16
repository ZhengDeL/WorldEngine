# Build the public post8 recipe first; supply its actual image tag or ID.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
COPY runtime/ /opt/worldengine-dependency-repair/
COPY wheelhouse/ /opt/worldengine-dependency-repair/wheelhouse/
RUN set -eu; \
    test "$(uname -m)" = aarch64; \
    if ! /opt/conda/envs/bevformer/bin/python -c "import ctypes; ctypes.CDLL('libGL.so.1'); ctypes.CDLL('libgthread-2.0.so.0')"; then \
      if command -v apt-get >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl1 libglib2.0-0; \
        rm -rf /var/lib/apt/lists/*; \
      elif command -v dnf >/dev/null 2>&1; then \
        dnf -y --setopt=install_weak_deps=False install 'libGL.so.1()(64bit)' 'libgthread-2.0.so.0()(64bit)'; \
        rpm -q --whatprovides 'libGL.so.1()(64bit)' 'libgthread-2.0.so.0()(64bit)'; \
        dnf clean all; \
      elif command -v yum >/dev/null 2>&1; then \
        yum -y install 'libGL.so.1()(64bit)' 'libgthread-2.0.so.0()(64bit)'; \
        rpm -q --whatprovides 'libGL.so.1()(64bit)' 'libgthread-2.0.so.0()(64bit)'; \
        yum clean all; \
      else \
        echo 'No supported package manager for required OpenCV libraries' >&2; exit 1; \
      fi; \
    fi; \
    /opt/conda/envs/bevformer/bin/python -c "import ctypes; ctypes.CDLL('libGL.so.1'); ctypes.CDLL('libgthread-2.0.so.0')"; \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/conda/envs/bevformer/bin/python \
      /opt/worldengine-dependency-repair/apply_minimal_repair.py

ENV PYTHONPATH=/opt/worldengine-dependency-fix:${PYTHONPATH}
COPY environment/ /opt/worldengine-environments/
COPY wheelhouse/ /opt/worldengine-environments/wheelhouse/
RUN set -eu; \
    test -d /opt/worldengine-dependency-fix; \
    test ! -e /opt/worldengine-sdk-tools311; \
    env -u PYTHONPATH -u PYTHONHOME /usr/bin/python3 -m venv --without-pip /opt/worldengine-sdk-tools311; \
    env PYTHONPATH=/opt/worldengine-environments/wheelhouse/pip-25.0.1-py3-none-any.whl \
      /opt/worldengine-sdk-tools311/bin/python -m pip install \
      --no-index --find-links /opt/worldengine-environments/wheelhouse --no-deps --require-hashes \
      --only-binary=:all: -r /opt/worldengine-environments/sdk-tools.lock; \
    /opt/conda/envs/bevformer/bin/python /opt/worldengine-environments/create_sdk_component_view.py; \
    test ! -e /opt/conda/envs/bevformer/lib/python3.10/site-packages/worldengine_application_environment.py; \
    test ! -e /opt/conda/envs/bevformer/lib/python3.10/site-packages/00_worldengine_application_environment.pth; \
    cp /opt/worldengine-environments/worldengine_application_environment.py /opt/conda/envs/bevformer/lib/python3.10/site-packages/; \
    printf '%s\n' 'import worldengine_application_environment' > /opt/conda/envs/bevformer/lib/python3.10/site-packages/00_worldengine_application_environment.pth; \
    chmod 0755 /opt/worldengine-environments/compiler_python311.sh

ENV HI_PYTHON=/opt/worldengine-environments/compiler_python311.sh
ENV PROJ_DATA=/opt/worldengine-geo/share/proj
ENV PROJ_LIB=/opt/worldengine-geo/share/proj
