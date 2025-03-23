FROM nvcr.io/nvidia/tritonserver:22.07-py3

# Copy model configuration if needed
COPY ./scripts/setup_triton.sh /setup.sh
RUN chmod +x /setup.sh

# Run setup script when container starts
ENTRYPOINT ["/setup.sh"]