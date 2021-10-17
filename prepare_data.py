import numpy as np
import urllib.request
import json
import csv

def parseData(fname):
    for l in urllib.request.urlopen(fname):
        yield eval(l)
    
def one_hot_year(y, max_year, min_year):
    ohv = np.zeros(max_year-min_year) # zero vector represents max_year
    if y - min_year == 0:
        return ohv
    else:
        ohv[y-min_year-1] = 1
    return ohv

def main():
    data = list(parseData("http://cseweb.ucsd.edu/classes/fa19/cse258-a/data/beer_50000.json"))

    prepared_data = {}
    feature_list = ['beer/ABV', 'review/text', 'review/timeStruct', 'review/appearance']
    target = 'review/overall'

    # process first feature
    ABV = [d[feature_list[0]] for d in data]
    print('ABV shape:', np.shape(ABV))
    # prepared_data[feature_list[0]] = ABV

    # process second feature
    review_length = [len(d[feature_list[1]]) for d in data]
    print('review length:', np.shape(review_length))
    # prepared_data[feature_list[1]] = review_length

    # process third feature
    years = [d[feature_list[2]]['year'] for d in data]
    min_year = np.min(years)
    max_year = np.max(years)
    ohvs = [one_hot_year(y, max_year, min_year) for y in years]
    print('ohvs shape:', np.shape(ohvs))
    # prepared_data['review/ohv_year'] = ohvs

    # process fourth feature
    appearance = [d[feature_list[3]] for d in data]
    print('appearance shape:', np.shape(appearance))
    # prepared_data[feature_list[3]] = appearance

    # process target
    y = [d['review/overall'] for d in data]
    print('y shape:', np.shape(y))

    Xy = [[a,b,c,d,e] for a,b,c,d,e in zip(ABV, review_length, ohvs, appearance, y)]
    
    with open('prepared_data.csv', 'w') as f:
    #     write_string = ''
    #     for entry in ['ABV', 'review_length', 'ohv_years', 'appearance', 'y']:
    #         write_string = write_string + entry + ', '
    #     f.write(write_string + '\n')

        for row in Xy:
            write_string = ''
            for entry in row:
                # print(np.shape(entry))
                if len(np.shape(entry))>0:
                    for e in entry:
                        write_string = write_string + str(e) + ', '
                else:
                    write_string = write_string + str(entry) + ', '

            f.write(write_string[:-2] + '\n')


    # with open('prepared_data.json', 'w') as f:
    #     json.dump(prepared_data, f)
        


if __name__ == '__main__':
    main()
    