import ctypes
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sys
import os


def parse_path(path):
    result = urlparse(path)
    return {
        'path': result[2],
        'query': dict((key, vals[0]) for key, vals in parse_qs(result[4]).items())
    }


class AgentLearningServer(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server,
                 activity_text_classifier, user_preferences_predictor):
        assert isinstance(activity_text_classifier, ActivityContentClassifier)
        assert isinstance(user_preferences_predictor, UserPreferencesPredictor)
        self.activity_text_classifier = activity_text_classifier
        self.user_preferences_predictor = user_preferences_predictor
        super().__init__(request, client_address, server)

    def _send_headers(self, state_code):
        self.send_response(state_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _respond(self, state_code, msg, data):
        self._send_headers(state_code)
        self.wfile.write(bytes(json.dumps({
            'message': msg,
            'data': data
        }), 'UTF-8'))

    def do_POST(self):
        params = parse_path(self.path)
        try:
            if params['path'] == '/classification':
                self.do_classification()
            elif params['path'] == '/preferences':
                self.do_preference_predicting()
        except Exception as err:
            self._respond(400, 'Error occurs: ' + str(err), None)

    def do_classification(self):
        ctype = self.headers.get('content-type')
        if ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers.get('content-length', 0))
            queries = parse_qs(self.rfile.read(length).decode('UTF-8'), keep_blank_values=1)
            print('Query: ', queries)
            if 'text' not in queries:
                self._respond(400, 'the param "text" should be given.', None)
            else:
                text = queries['text']
                activity_tags = self.activity_text_classifier.classify(text)
                self._respond(200, 'success', activity_tags)
        else:
            self._respond(400, 'the content type should be set "content-type".', None)

    def do_preference_predicting(self):
        ctype = self.headers.get('content-type')
        if ctype == 'application/json':
            content_len = int(self.headers.get('content-length', 0))
            body = json.loads(self.rfile.read(content_len).decode('UTF-8'))
            if 'user' not in body:
                self._respond(400, 'the param "user" should be given.', None)
            elif 'association_histories' not in body:
                self._respond(400, 'the param "association_histories" should be given.', None)
            elif 'target_activity' not in body:
                self._respond(400, 'the param "target_activity" should be given.', None)
            else:
                user = body['user']
                target_activity = body['target_activity']
                association_histories = body['association_histories']
                possibility = self.user_preferences_predictor.get_possibility(user,
                                                                              association_histories,
                                                                              target_activity)
                print('Possibility: ', possibility)
                self._respond(200, 'success', {'user': user, 'activity': target_activity, 'possibility': possibility})
        else:
            self._respond(400, 'The content type should be set "application/json".', None)


def run(server_class=HTTPServer, handler_class=None, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Now the http server is serving.')
    httpd.serve_forever()


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


if __name__ == "__main__":
    from base import *
    from stubs import *
    from keras_model_adapters import *
    from sys import argv

    # if the script is not run in the administrator privilege, pop up the UAC dialog to seek to elevate
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

    print('Models are building...')
    activity_text_classifier = StubActivityContentClassifier()  # KerasActivityContentClassifier()
    user_preferences_predictor = StubUserPreferencesPredictor()  # KerasUserPreferencesPredictor()

    print('Models built successfully.')

    port = int(argv[1]) if len(argv) == 2 else 5000

    run(port=port, handler_class=lambda request, client_address, server: AgentLearningServer(request, client_address, server, activity_text_classifier, user_preferences_predictor))

