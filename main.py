import csdmd.webcam as webcam
import os
import click

@click.command()
@click.option('-s', '--source', 'src', default='webcam0', help="path of video file or name of device you want to parse, default is 'webcam0'")  #
@click.option('-d', '--destination', 'dest', default='')
def main(src, dest):

    if src== 'webcam':
        src = 0
    if src[:6] == 'webcam':
        src = int(src[6:])
    else:
        assert (os.path.exists(src)), "The given source file does not exist"
    
    
    if dest == '':
        dest = None
    elif not os.path.exists(dest):
        os.mkdir(dest)


    webcam.run(src, dest)

if __name__ == "__main__":
    webcam.run(src=0,dest='data/kleintyv/IMG_0453.MOV') 

