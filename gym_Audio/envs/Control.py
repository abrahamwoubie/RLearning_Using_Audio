import numpy as np
import aubio
nrow=4
ncol=4
pitch={}
s={}
state=[]
pitch={}
new_state=[]

def Extract_Pitch(row, col):
    pitch_List = []
    sample_rate = 44100
    x = np.zeros(44100)
    for i in range(44100):
        x[i] = np.sin(2. * np.pi * i * 225. / sample_rate)

    # create pitch object
    p = aubio.pitch("yin", samplerate=sample_rate)

    # pad end of input vector with zeros
    pad_length = p.hop_size - x.shape[0] % p.hop_size
    x_padded = np.pad(x, (0, pad_length), 'constant', constant_values=0)
    # to reshape it in blocks of hop_size
    x_padded = x_padded.reshape(-1, p.hop_size)

    # input array should be of type aubio.float_type (defaults to float32)
    x_padded = x_padded.astype(aubio.float_type)

    for frame, i in zip(x_padded, range(len(x_padded))):
        time_str = "%.2f" % (i * p.hop_size / float(sample_rate))
        pitch_candidate = p(frame)[0] + row + col
        # print(pitch_candidate)
        pitch_List.append(pitch_candidate)
    return pitch_List

def find_row_col(row, col,action):
    if action == 0:  # left
        col = max(col - 1, 0)
    elif action == 1:  # down
        row = min(row + 1, nrow - 1)
    elif action == 2:  # right
        col = min(col + 1, ncol - 1)
    elif action == 3:  # up
        row = max(row - 1, 0)
    return (row, col)


if __name__ == '__main__':
    # a = []
    # dist={}
    # for i in range(0,nrow):
    #     for j in range(0,ncol):
            #pitch[i,j] = i*4+j

    for i in range(0,nrow):
        for j in range(0,ncol):
            for action in range(0,4):
                r,c=find_row_col(i, j,action)
                pitch[i,j]=Extract_Pitch(i,j)
                s[i,j]=i*nrow+j
                #s[i,j]=i*nrow+j
                #print('Row {} Column {} Action {} new_Row {} new_Column {}'.format(i,j,action,r,c))

    for i in range(0,nrow):
        for j in range(0,ncol):
            #print("i {} j {}".format(i,j))
            if (i!=nrow-1):
                down=sum(pitch[i+1,j])
                #down=pitch[i+1,j]
            else:
                down = sum(pitch[i, j])
                #down = pitch[i,j]
            if (j!=ncol-1):
                right=sum(pitch[i,j+1])
                #right = pitch[i,j+1]
            else:
                right = sum(pitch[i, j])
                #right = pitch[i,j]

            if(down>=right):
                current=down
                #state.append(current)
                state.append(current)
                new_state.append(i*nrow+j)
                break
            else:
                current=right
                new_state.append(i * nrow + j)
                state.append(current)
    print(state)
    print(new_state)


    for i in range(0,nrow):
        for j in range(0,ncol):
            #print(sum(pitch[i,j]))
            pass