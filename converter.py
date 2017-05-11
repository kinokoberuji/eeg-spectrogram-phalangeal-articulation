import csv
import os
import sys

def convert_txt_to_csv(input_path, output_path):
  """ Converts text format EEG data into CSV format
  """
  input_sample_rate = 250

  input_headers = ['id','chan1','chan2','chan3','chan4','chan5','chan6','chan7','chan8','accel1','accel2','accel3'];
  output_headers = ['Time','chan1','chan2','chan3','chan4','chan5','chan6','chan7','chan8','Sample_rate']

  files = os.listdir(input_path)

  txt_files = []
  for filename in files:
    if '.txt' in filename:
      txt_files.append(filename)

  if(len(txt_files) < 1):
    print "ERROR: No input files found! \n Input files must be placed in the '"+input_path+"' directory  (relative \n to  the location of this script), or change the 'input_path'\n parameter in the script"
    raw_input()
    
  for input_fn in txt_files:

    print "- {0}".format(input_fn)

    output_data = []
    time_counter = 0
    time_increment = float(1)/float(input_sample_rate)

    with open(os.path.join(input_path,input_fn), 'rb') as txtfile:

       for i, line in enumerate(txtfile):
            if i == 2:
              sr_line = line
              break

       txt_input = csv.DictReader(txtfile, fieldnames=input_headers, dialect='excel')
       row_count = 0

       for row in txt_input:

            row_count = row_count + 1

            if(row_count > 2):

              output = {}

              time_counter = time_counter + time_increment

              output['Time'] = time_counter

              for i in range(1,9):
                channel_key = 'chan'+str(i)
                output[channel_key] = row[channel_key]

              output['Sample_rate'] = input_sample_rate

              output_data.append(output)

    output_fn = input_fn.replace(".txt", ".csv")

    with open(os.path.join(output_path,output_fn), 'wb') as output_csv_file:

      csv_output = csv.DictWriter(output_csv_file, fieldnames=output_headers, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

      headers_text = {}

      for val in output_headers:
        headers_text[val] = val

      csv_output.writerow(headers_text)

      for row in output_data:
        csv_output.writerow(row)

