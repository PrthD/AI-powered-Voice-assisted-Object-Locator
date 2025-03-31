from datetime import datetime
import pandas

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

