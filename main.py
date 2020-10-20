import csdmd.player as player
import os
import click


@click.group()
def main():
    pass


@main.command()
@click.option('-s', '--source', 'src', default='webcam0', help="path of video file or name of device you want to parse, default is 'webcam0'")  #
def play(src):

    if src== 'webcam':
        src = 0
    if src[:6] == 'webcam':
        src = int(src[6:])
    else:
        assert (os.path.exists(src)), "The given source file does not exist"

    player.run(src)


@main.command()
@click.option('-s', '--source', 'src', default='webcam0', help="path of video file or name of device you want to parse, default is 'webcam0'")  #
@click.option('-d', '--destination', 'dest', default='out', help="path where you want to put the output files, default is 'out'")
def process(src, dest):
    import csdmd.process as process
    if src== 'webcam':
        src = 0
    if src[:6] == 'webcam':
        src = int(src[6:])
    else:
        assert (os.path.exists(src)), "The given source file does not exist"
    
    if not os.path.exists(dest):
        os.mkdir(dest)

    process.run(src, dest)

    
# This is just used for debugging,
if __name__ == "__main__":
    player.run(src='data/kleintyv/IMG_0453.MOV') 

