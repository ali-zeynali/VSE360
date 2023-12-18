import os
import shutil
import subprocess
import time
import math
import numpy as np
import pychrome
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


class ChromeDownloader:
    def __init__(self, port=7000):
        self.port = port
        self.server_path = r"C:\Users\fredb\ServerDir\\"
        self.client_path = r"C:\Users\fredb\ClientDir\Video\\"
        print("Initializing the downloader")
        self.free_port(self.port)
        self.find_port()

    def free_port(self, port, __print__=False):
        cnt = 0
        while True:
            try:
                find_pid = subprocess.run('netstat -ano -p tcp|find \"{0}\"'.format(port), shell=True, check=True,
                                          stdout=subprocess.PIPE)
            except Exception as e:
                # print(e)
                if __print__:
                    print("{0} process got killed".format(cnt))
                return
            find_pid = str(find_pid.stdout, "UTF-8").split("\n")
            if len(find_pid) > 0:
                find_pid = find_pid[0].replace("\r", "").split(" ")
            if len(find_pid) > 1:
                pid = int(find_pid[-1])
                if pid == 0:
                    return
                try:
                    killing_pid = subprocess.run('taskkill /PID {0} /F'.format(pid), shell=True, check=True,
                                                 stdout=subprocess.PIPE)
                except Exception as e:
                    print("[From free_port] ", str(e))
                    return
                cnt += 1

    def find_port(self):

        port = self.port
        while True:
            try:
                options = webdriver.ChromeOptions()
                prefs = {"download.default_directory": self.client_path, "safebrowsing.enabled": "false"}
                options.add_experimental_option("prefs", prefs)
                options.headless = True
                options.add_argument("--remote-debugging-port={0}".format(port))
                options.add_argument("--disk-cache-size={0}".format(0))
                driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
                self.driver = driver

                dev_tools = pychrome.Browser(url="http://localhost:{0}".format(port))
                # dev_tools = pychrome.Browser(url="http://localhost")
                tab = dev_tools.list_tab()[0]
                tab.start()
                break
            except Exception as e:
                port += 1
        print("Downloading port switched to {0}".format(port))
        self.port = port

    def remove_file(self, path, __print__=False):
        time.sleep(0.5)
        try:
            os.remove(path)
        except Exception as e:
            if __print__:
                print("[From remove_file]: ", e)
            else:
                pass
        return

    def remove_directory(self, path, __print__=False):
        time.sleep(0.5)
        try:
            shutil.rmtree(path)
        except Exception as e:
            if __print__:
                print("[From remove_directory]: ", e)
            else:
                pass
        return

    def make_directory(self, path, __print__=False):
        self.remove_directory(path, __print__=__print__)
        try:
            os.mkdir(path)
        except Exception as e:
            if __print__:
                print("[From make_directory]: ", e)
            else:
                pass
        return

    def write_to_file(self, text, path):
        with open(path, 'w') as writer:
            writer.write(text)

    def make_file(self, size_in_bytes, path):
        size = int(math.ceil(size_in_bytes))
        text = size * "A"
        self.write_to_file(text, path)

    def quit(self):
        self.driver.quit()

    def download_file(self, file_size, bandwidth_capacity, __print__=False):
        """

        :param file_size: size of requested file in bytes
        :param bandwidth_capacity: bandwidth capacity in bytes per second
        :return:
        """

        download_id = int(np.random.random() * 1.e9)

        # debug_port = 8080

        while True:
            try:

                dev_tools = pychrome.Browser(url="http://localhost:{0}".format(self.port))
                # dev_tools = pychrome.Browser(url="http://localhost")
                tab = dev_tools.list_tab()[0]
                tab.start()

                tab.call_method("Network.emulateNetworkConditions",
                                offline=False,
                                latency=100,
                                downloadThroughput=int(bandwidth_capacity),
                                uploadThroughput=int(bandwidth_capacity),
                                connectionType="cellular3g")

                file_name = "_{0}".format(download_id)
                self.make_file(file_size, self.server_path + file_name)
                server_size = os.path.getsize(self.server_path + file_name)

                self.remove_file(self.client_path + file_name, __print__=__print__)
                start = time.time()

                # driver.get(test_page)
                time.sleep(1)
                self.driver.get("http://127.0.0.1:8888/ServerDir/" + file_name)
                cnt = 0
                while True:
                    try:
                        client_size = os.path.getsize(self.client_path + file_name)
                        if client_size >= server_size:
                            break
                    except Exception as e:
                        time.sleep(0.001)
                        cnt += 1
                # print("Client file size is: {0}, after {1} sleep".format(client_size, cnt))
                # print("Slowed get time: {0}".format(time.time() - start))
                self.remove_file(self.server_path + file_name, __print__=__print__)
                self.remove_file(self.client_path + file_name, __print__=__print__)
                # try:
                #     driver.close()
                # except Exception as e:
                #     pass
                # print("Port {0} got used for download".format(self.port))
                end_time = time.time()
                return end_time - start
            except Exception as e:
                # print("Port {0} was busy".format(self.port))

                self.free_port(self.port, __print__=__print__)
                self.find_port()
                print("Port was busy")
                print(e)
                return

    # print(download_file(10 ** 7, 2 * 10 ** 6))
