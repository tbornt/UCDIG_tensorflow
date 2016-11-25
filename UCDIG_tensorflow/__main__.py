import os
import argparse
import tensorflow as tf
from model.dtn import DTN


class Env:
    pass


ENV = Env()


def train(args):
    with tf.Session() as sess:
        dtn = DTN(sess)
        dtn.train()


def test(args):
    with tf.Session() as sess:
        dtn = DTN(sess)
        dtn.test()


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)


def init_env():
    ENV.proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV.data_dir = os.path.join(ENV.proj_dir, 'data')
    if not os.path.exists(ENV.data_dir):
        os.mkdir(ENV.data_dir)
    ENV.sample_dir = os.path.join(ENV.proj_dir, 'sample')
    if not os.path.exists(ENV.sample_dir):
        os.mkdir(ENV.sample_dir)


def main():
    init_env()
    parse_args()


if __name__ == '__main__':
    main()
