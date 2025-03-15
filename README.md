# Image Colorization using Deep Learning

This project colorizes black-and-white images using a pre-trained deep learning model based on OpenCV's DNN module. The model has been trained on ImageNet using the Caffe framework.

## Project Structure

```
Colorize_Image-Using_DEEP_LEARNING/
│-- models/
│   ├── colorization_release_v2.caffemodel  # Downloaded automatically
│   ├── colorization_deploy_v2.prototxt    # Download manually
│-- gui.py                                  # Tkinter-based GUI
│-- colorize.py                             # Script for colorization
│-- requirements.txt                        # Dependencies
│-- README.md                               # Project documentation
│--pts_in_hull.npy                          # Download manually
```

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python installed (preferably Python 3.8+). Install required libraries using:

```sh
pip install -r requirements.txt
```

### 2. Download Model Files

Create a `models` directory in the project root and place the required model files inside it.

#### **Automatically Download**

Run the following Python script to download the `.caffemodel` file automatically:

```python
import requests

url = "https://github.com/richzhang/colorization/raw/master/colorization/models/colorization_release_v2.caffemodel"
output_path = "models/colorization_release_v2.caffemodel"

print("Downloading model...")
response = requests.get(url, stream=True)
with open(output_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)
print("Download complete!")
```

#### **Manually Download Other Required Files**

- **Prototxt file**: [Download here](https://github.com/richzhang/colorization/blob/master/colorization/models/colorization_deploy_v2.prototxt)
- **Numpy file**: [Download here](https://github.com/richzhang/colorization/blob/master/colorization/models/pts_in_hull.npy)

Place these files inside the `models/` directory.

### 3. Run the GUI Application

Once all dependencies and models are set up, run the GUI using:

```sh
python gui.py
```

This will open a Tkinter-based interface to upload and colorize black-and-white images.

## Usage

- Click **Upload Image** to select a black-and-white image.
- Click **Colorize** to generate a colorized version of the image.
- Click **Save** to save the output image.

## Example Output

| Input (B&W) | Output (Colorized) |
| ----------- | ------------------ |
|![image other](https://github.com/user-attachments/assets/b08167ea-cf1d-43ef-b5b2-8fb3bfc0b8e8)
|![result](https://github.com/user-attachments/assets/d99b0c1f-eba9-4cb0-9590-b0ae0a26d339)
|

## References

- [Rich Zhang’s Colorization Research](https://richzhang.github.io/colorization/)
- [OpenCV DNN Documentation](https://docs.opencv.org/master/d6/d0f/group__dnn.html)

## License

This project is for educational purposes. Model weights and prototxt files are from [Richard Zhang’s research](https://github.com/richzhang/colorization/).
