from http.server import HTTPServer, BaseHTTPRequestHandler

from io import BytesIO
from deepspeech import Model
import scipy.io.wavfile as wav
 

modelparams = {
    "model" :"models/output_graph.pbmm",
    "alphabet": "models/alphabet.txt",
    "lm": "models/lm.binary",
    "trie": "models/trie",
    "n_features": 26,
    "n_context": 9,
    "beam_width": 500,
    "lm_alpha": 0.75,
    "lm_beta": 1.85
}

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = BytesIO(self.rfile.read(content_length))
        self.send_response(200)
        self.end_headers()
        fs, audio = wav.read(body)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        text = ds_model.stt(audio, fs)
        print("STT result: {}".format(text))
                           
        response = BytesIO()
        #response.write(b'This is POST request. ')
        #response.write(b'Received: ')
        response.write(text.encode("utf-8"))
        self.wfile.write(response.getvalue())

def setup_model():
    ds_model = Model(
        modelparams["model"], modelparams["n_features"], modelparams["n_context"],  modelparams["alphabet"], modelparams["beam_width"])
    ds_model.enableDecoderWithLM(modelparams["alphabet"], modelparams["lm"], modelparams["trie"], modelparams["lm_alpha"], modelparams["lm_beta"])
    print("model")
    return ds_model


ds_model = setup_model() 
httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
