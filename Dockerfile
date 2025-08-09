FROM tensorflow/tensorflow
RUN mkdir -p /program
WORKDIR /program
COPY requirements.txt /program/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /program/app.py
# Install JupyterLab
RUN pip install jupyterlab

# Expose Jupyter port
EXPOSE 8888

# Run JupyterLab when container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]