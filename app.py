from flask import Flask, render_template, request
import torch
from model import CNN
from PIL import Image
from torchvision import transforms
import webbrowser

app = Flask(__name__)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            try:
                img = Image.open(file).convert("L")
                x = transform(img).unsqueeze(0)

                with torch.no_grad():
                    out = model(x)
                    prob = torch.softmax(out, dim=1)
                    conf, pred = prob.max(1)

                result = f"{CLASS_NAMES[pred.item()]} ({conf.item()*100:.2f}%)"

            except Exception as e:
                result = f"Error: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)