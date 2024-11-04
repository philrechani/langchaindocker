import requests
import json
import subprocess
import time
import os
import signal
import platform

class LlamaAPI:
    def __init__(self) -> None:
        self.context = None
        self.host = "http://localhost:8080/"
        self.port = '8080'
        self.exe_path = ''
        self.headers = {
                            "Content-Type": "application/json"
                        }
        self.is_completion = True
        self.model_path = ''
        self.process = None
        
    def url(self,host,endpoint):
        return f'{host}{endpoint}'
    
    def set_model_path(self,path):
        self.model_path = path
        
    def set_exe_path(self,path):
        self.exe_path = path
    
    def initialize_server(self, type='completion', docker=False):
        if self.process:
            print('Shutting down previous server')
            try:
                # On Windows, we need to kill the entire process group
                if platform.system() == 'Windows':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for up to 5 seconds for the process to end
                for _ in range(50):  # 50 * 0.1 seconds = 5 seconds
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.1)
                else:
                    # If it's still running after 5 seconds, force kill
                    print("Process didn't terminate, forcing kill...")
                    if platform.system() == 'Windows':
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                    else:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
            except Exception as e:
                print(f"Error shutting down previous server: {e}")
        
        server_executable = 'llama-server.exe'
        port = self.port
        n_gpu_layers = 99
        ctx_size = 2048
        ubatch_size = 2048
        batch_size = 4096
        command = [
            self.exe_path + server_executable,
            '-m', self.model_path,
            '--port', str(port),
            '--n-gpu-layers', str(n_gpu_layers),
            '--ctx_size', str(ctx_size)
        ]
        
        if type != 'completion':
            command.append('--embedding')
        
        # Open the subprocess in a new shell
        if platform.system() == 'Windows':
            self.process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            self.process = subprocess.Popen(command, preexec_fn=os.setsid, start_new_session=True)
        
        # Wait a bit to ensure the server has started
        time.sleep(2)
        
        # Check if the process is running
        if self.process.poll() is None:
            print("Server started successfully")
        
        return self.process
        
    
    def tokenize(self, 
                 content, 
                 endpoint = 'tokenize',
                 with_pieces = True):
        
        data = {
            'content': content,
            'with_pieces': with_pieces
        }
        
        response = requests.post(self.url(self.host,endpoint), 
                                 headers = self.headers,
                                 json = data)
        return response.json()
    
    def detokenize(self):
        pass
    
    def completion(self,prompt, n_predict = 2048, endpoint = 'completion'):
        
        data = {
            'prompt': prompt,
            'n_predict': n_predict
        }
        response = requests.post(self.url(self.host, endpoint), 
                                 headers = self.headers,
                                 json = data)
        return response.json()
    
    def health():
        pass
    
    def embedding(self, content, endpoint='embedding'):
        data = {
            'content': content
        }
        
        response = requests.post(self.url(self.host, endpoint), 
                                 headers = self.headers,
                                 json = data)
        return response.json()
    
