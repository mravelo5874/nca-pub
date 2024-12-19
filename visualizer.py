from http.server import HTTPServer, SimpleHTTPRequestHandler
from colorama import init, Style, Fore
from argparse import Namespace
from scripts.nca import utils
import os

init()
PROGRAM = f'{Style.DIM}[{os.path.basename(__file__)}]{Style.RESET_ALL}'
TIMESTAMP = None

def main(args: Namespace) -> None:
    # start server to host visualizer
    print (f'{PROGRAM} starting local server on: {Fore.CYAN}http://localhost:{args.port}{Style.RESET_ALL}')
    os.chdir('./scripts/visualizer/build/')
    httpd = HTTPServer(('localhost', args.port), SimpleHTTPRequestHandler)
    httpd.serve_forever()

if __name__ == '__main__':
    args = utils.parse_visualizer_args()
    main(args)