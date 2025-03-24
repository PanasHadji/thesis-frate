import matplotlib.pyplot as plt
import io
import base64

class ImageEncoder:
    def __init__(self):
        print('ImageEncoder!')

    def encode_plt_into_base64(self, plt):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode('utf-8')
        print(encoded)

        html_img = f'<img src="data:image/png;base64,{encoded}"/>'
        print(html_img)
        return html_img