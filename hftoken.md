If you have a Hugging Face account, we should absolutely use it to authenticate. That way you can pull down the cutting-edge v2.5 models rather than settling for the older public ones.

Because Ray spawns background worker processes, the cleanest way to pass your credentials to all the actors is by authenticating your entire local environment using the Hugging Face CLI.

1. **Generate an Access Token:**
1. Log in to [Hugging Face](https://huggingface.co/).
2. Go to **Settings** > **Access Tokens** (or [click here](https://huggingface.co/settings/tokens)).
3. Click **Create new token**, name it (e.g., `macbook-ray`), and set the type to **Read**.
4. Copy the generated token.


2. **Authenticate your terminal:**
In your active `acf_venv` terminal, install the HF Hub (if it isn't already) and log in.

```bash
pip install huggingface_hub
huggingface-cli login

```

When prompted, paste your token. It will save the credential to `~/.cache/huggingface/token`, which the GLiNER library will automatically detect when the Ray actors boot up.


3. **Accept the Model Agreement:** Required for gated repositories.
Many domain-specific models require you to manually click an agreement before your API token is allowed to download them.

1. Go to the URL that threw the error: [https://huggingface.co/gliner-community/gliner_base-v2.5](https://huggingface.co/gliner-community/gliner_base-v2.5) *(Note: if that throws a 404, the correct repository name might be `gliner-community/gliner-v2.5` or `urchade/gliner_multi-v2.5` depending on the exact Fastino release)*.
2. If there is a banner asking you to agree to the terms, click **Agree and access repository**.


4. **Rerun the Pipeline:**
Now that your environment is authenticated and the gated access is approved, run the pipeline again.

```bash
python ray_pipeline.py

```


Ray will boot up, detect your cached Hugging Face token, successfully download the v2.5 tensor weights into your Mac's GPU unified memory, and process the batch of 8 reports.