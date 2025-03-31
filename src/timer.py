from datetime import datetime
import pandas as pd

class Timer:
  def __init__(self) -> None:
         self.time_dict = {"st": {"start": None,
                                  "end": None},
                           "d": {"start": None,
                                  "end": None},
                           "ts": {"start": None,
                                  "end": None},
                           "time": None}
  

  def log_start(self,
                name: str) -> bool:
                start_time = datetime.now()
                self.time_dict[name]["start"] = start_time.microsecond
                if self.time_dict[name]["start"] == None:
                  return False
                
                else:
                  return True

  def log_end(self,
              name: str) -> bool:
              end_time = datetime.now()
              self.time_dict[name]["end"] = end_time.microsecond
              if self.time_dict[name]["end"] == None:
                return False
              
              else:
                return True

  def clean_start(self,
                  name: str) -> bool:
                  self.time_dict[name]["start"] = None
                  if self.time_dict[name]["start"] == None:
                    return True
                  
                  else:
                    return False

  def clean_end(self,
                name: str) -> bool:
                self.time_dict[name]["end"] = None
                if self.time_dict[name]["end"] == None:
                  return True
                
                else:
                  return False

  def new_session(self):
    self.time_dict["time"] = datetime.now().strftime("%H:%M:%S")
    self.time_dict["st"] = {"start": None,
                            "end": None}
    self.time_dict["d"] = {"start": None,
                           "end": None}
    self.time_dict["ts"] = {"start": None,
                            "end": None}
  
  def save_log(self)-> bool:
    if self.check_log_completeness() == False:
      return False
    data = pd.read_excel("time_statistics.xlsx")
    new_field = pd.DataFrame(columns = ['Session', 'Speech-to-Text', 'Object Detection', 'Text-to-Speech',
 'Total Time'])
    new_field['Session'] = self.time_dict["time"]
    new_field['Speech-to-Text'] = self.time_dict["st"]["end"] - self.time_dict["st"]["start"]
    new_field['Object Detection'] = self.time_dict["d"]["end"] - self.time_dict["d"]["start"]
    new_field['Text-to-Speech'] = self.time_dict["ts"]["end"] - self.time_dict["ts"]["start"]
    new_field['Total Time'] = new_field['Speech-to-Text'] + new_field['Object Detection'] + new_field['Text-to-Speech']
    data = pd.concat([data, new_field], ignore_index = True)
    data.to_excel("time_statistics.xlsx", index = False)
    return True

  def check_log_completeness(self):
    if self.time_dict["st"]["start"] == None:
      return False
    elif self.time_dict["st"]["end"] == None:
      return False
    elif self.time_dict["d"]["start"] == None:
      return False
    elif self.time_dict["d"]["end"] == None:
      return False
    elif self.time_dict["ts"]["start"] == None:
      return False
    elif self.time_dict["ts"]["end"] == None:
      return False
    else:
      return True
    

# if __name__ == '__main__':
#     timer = Timer()
#     timer.save_log()
