import os
from urllib.request import urlretrieve
import tarfile
from glob import glob
import argparse
import zipfile

import src.config

def download_chest_xray(path=src.config.directories['chestx-ray8']):
    if not os.path.exists(path):
        os.mkdir(path)

    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    for idx, link in enumerate(links):
        print('Downloading', link, '...')
        filename = f'images_{idx:01}.tar.gz'
        filepath = os.path.join(path, filename)
        urlretrieve(link, filepath)
    print('Download complete')

    file_pattern = os.path.join(path, 'images_*.tar.gz')
    for fp in glob(file_pattern):
        print(f'Extracting {fp}')
        tar = tarfile.open(fp)
        tar.extractall(path)
        tar.close()
    print('Extraction complete')


def download_chaos(path=src.config.directories['chaos']):
    if not os.path.exists(path):
        os.mkdir(path)

    links = [
        'https://zenodo.org/record/3431873/files/CHAOS_Train_Sets.zip?download=1',
        'https://zenodo.org/record/3431873/files/CHAOS_Test_Sets.zip?download=1'
    ]

    for idx, link in enumerate(links):
        if 'CHAOS_Train_Sets.zip' in link:
            filename = 'CHAOS_Train_Sets.zip'
        else:
            filename = 'CHAOS_Test_Sets.zip'

        print('Downloading', link, '...')
        filepath = os.path.join(path, filename)
        urlretrieve(link, filepath)
    print('Download complete')

    file_pattern = os.path.join(path, '*.zip')
    for fp in glob(file_pattern):
        print(f'Extracting {fp}')
        archive = zipfile.ZipFile(fp)
        archive.extractall(path)
        archive.close()
    print('Extraction complete')


def download_lits(path=src.config.directories['lits']):
    raise NotImplementedError


def download_pulmonary_cxr_abnormalities(path=src.config.directories['pulmonary_cxr_abnormalities']):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='download.py')
    parser.add_argument('dataset', 
        help='The dataset to download', 
        choices=['chestx-ray8', 'lits', 'chaos', 'pulmonary-cxr-abnormalities', 'all'])

    args = parser.parse_args()

    if args.dataset == 'chestx-ray8':
        download_chest_xray()
    elif args.dataset == 'chaos':
        download_chaos()
    elif args.dataset == 'lits':
        download_lits()
    elif args.dataset == 'pulmonary-cxr-abnormalities':
        download_pulmonary_cxr_abnormalities()
    elif args.dataset == 'all':
        download_chest_xray()
        download_chaos()
        download_lits()
        download_pulmonary_cxr_abnormalities()
