# Installing Google Cloud SDK on macOS

This guide will help you install and set up the Google Cloud SDK, which is required for deploying the OPT-RAG application to Google Kubernetes Engine (GKE).

## Installation Steps

### 1. Install Google Cloud SDK

The easiest way to install Google Cloud SDK on macOS is using Homebrew:

```bash
# Update Homebrew
brew update

# Install Google Cloud SDK
brew install --cask google-cloud-sdk
```

If you don't use Homebrew, you can install it manually:

1. Download the SDK from: https://cloud.google.com/sdk/docs/install-sdk

2. Extract the archive to a location on your system:
   ```bash
   tar -xvf google-cloud-sdk-*.tar.gz
   ```

3. Run the installation script:
   ```bash
   ./google-cloud-sdk/install.sh
   ```

4. Restart your terminal or source your shell configuration file:
   ```bash
   source ~/.zshrc  # or ~/.bash_profile depending on your shell
   ```

### 2. Verify Installation

Verify that the Google Cloud SDK is properly installed:

```bash
gcloud --version
```

You should see version information for Google Cloud SDK.

### 3. Initialize the SDK

Initialize the Google Cloud SDK:

```bash
gcloud init
```

This will:
1. Ask you to log in with your Google account
2. Let you select a default GCP project
3. Configure a default compute region and zone

### 4. Install Components

Install additional components:

```bash
# Install kubectl for Kubernetes management
gcloud components install kubectl

# Install other useful components
gcloud components install beta
```

### 5. Configure Docker for GCR

Configure Docker to use Google Cloud credentials for pushing images:

```bash
gcloud auth configure-docker
```

### 6. Set Default Project

Set your default project:

```bash
gcloud config set project YOUR_PROJECT_ID
```

### 7. Enable Required APIs

Enable the necessary APIs for your project:

```bash
# Enable Kubernetes Engine API
gcloud services enable container.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com
```

## Next Steps

After installing the Google Cloud SDK, you can proceed with deploying the OPT-RAG application to GKE:

1. Build and push Docker images
2. Create a GKE cluster
3. Deploy the application

Refer to the main Kubernetes deployment guide for detailed instructions.

## Troubleshooting

If you encounter any issues with the installation:

- **Path Issues**: Make sure that the Google Cloud SDK is in your PATH. Add the following to your `~/.zshrc` or `~/.bash_profile`:
  ```bash
  source "$(brew --prefix)/share/google-cloud-sdk/path.zsh.inc"
  source "$(brew --prefix)/share/google-cloud-sdk/completion.zsh.inc"
  ```

- **Authentication Issues**: If you're having problems with authentication, try:
  ```bash
  gcloud auth login
  ```

- **Proxy Settings**: If you're behind a proxy, configure it with:
  ```bash
  gcloud config set proxy/type http
  gcloud config set proxy/address PROXY_ADDRESS
  gcloud config set proxy/port PROXY_PORT
  ```

- **Component Installation Issues**: If you have problems installing components:
  ```bash
  sudo gcloud components update
  sudo gcloud components install COMPONENT_NAME
  ``` 