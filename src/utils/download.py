import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup as BSoup

def download_mitdb():
    """ All """
    extensions = ['atr', 'dat', 'hea']
    the_path = 'https://www.physionet.org/physiobank/database/mitdb/'

    # Save to proper data/ directory
    savedir = '/home/carlitos/Documents/Projects/ecg_classification/data/raw/mitdb'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # With this format
    savename = savedir + '/{}.{}'

    # Find all interesting files on that site:
    soup = BSoup(requests.get(the_path).text, 'html.parser')

    # Find all links pointing to .dat files
    hrefs = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Download datafiles with markers given
        if href.endswith('.dat'):
            hrefs.append(href[:-4])

    # Path to the file on the internet
    down_path = the_path + '{}.{}'

    for data_id in hrefs:
        for ext in extensions:
            webpath = down_path.format(data_id, ext)
            datafile = requests.get(webpath, stream=True)

            # Save locally
            filepath = savename.format(data_id, ext)
            with open(filepath, 'wb') as out:
                for chunk in tqdm(datafile.iter_content(chunk_size=1024)):
                    if chunk:
                        out.write(chunk)

    print('Downloaded {} data files'.format(len(hrefs)))

def download_qt():
    """ All """
    extensions = ['atr', 'dat', 'hea',
                  'man', 'q1c', 'q2c',
                  'qt1', 'qt2', 'pu', 'pu0', 'pu1']
    the_path = 'https://www.physionet.org/physiobank/database/qtdb/'

    # Save to proper data/ directory
    savedir = './data/raw/qt'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # With this format
    savename = savedir + '/{}.{}'

    # Find all interesting files on that site:
    soup = BSoup(requests.get(the_path).text, 'html.parser')

    # Find all links pointing to .dat files
    hrefs = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Download datafiles with markers given
        if href.endswith('.dat'):
            hrefs.append(href[:-4])

    # Path to the file on the internet
    down_path = the_path + '{}.{}'

    for data_id in hrefs:
        for ext in extensions:
            webpath = down_path.format(data_id, ext)
            try:
                datafile = requests.get(webpath, stream=True)

                # Save locally
                filepath = savename.format(data_id, ext)
                with open(filepath, 'wb') as out:
                    for chunk in tqdm(datafile.iter_content(chunk_size=1024)):
                        if chunk:
                            out.write(chunk)

            # Assuming that 404 (Not Found)
            # is the only one possible http error
            except requests.HTTPError:
                print('Not available:', webpath)

    print('Downloaded {} data files'.format(len(hrefs)))

if __name__ == '__main__':
    download_mitdb()
