"""Optional, explicitly invoked Colab Drive integration."""


def mount_drive(force_remount=True):
    """Mount Google Drive in Colab. Importing this module has no side effects."""
    from google.colab import drive

    drive.mount("/content/drive", force_remount=force_remount)
    return "/content/drive/MyDrive/openai-gsm8k"
