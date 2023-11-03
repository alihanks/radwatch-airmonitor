import subprocess
import datetime
import smtplib
import os

run_time=3.25
three_days_from_now=datetime.datetime.now()+datetime.timedelta(days=run_time)
stop_acq_time=three_days_from_now+datetime.timedelta(days=run_time/3.)
warn_acq_time=three_days_from_now+datetime.timedelta(days=run_time/5.)

def send_mail(msg_):
# Needs to have a try except for whole function
    msg_='\n'+msg_
    smtpserver=smtplib.SMTP("smtp.gmail.com",587)
    subject='Air Monitor Automail'
    to_='jahanks@berkeley.edu'
    from_='ucb.cram@gmail.com'
    header='To:'+to_+'\n'+'From:'+from_+'\n'+'Subject'+subject+'\n'
    try:
        smtpserver.ehlo()
        smtpserver.starttls()
        smtpserver.login('ucb.cram@gmail.com','electron374keV')
        smtpserver.sendmail(from_,to_,msg_)
    except smtplib.SMTPHeloError:
        print("Hello error from SMTP")
    except smtplib.SMTPAuthenticationError:
        print("Could not authenticate")
    except smtplib.SMTPException:
        print("Could not send mail!!!")
    except :
        print("Could not connect to internet")
        
    smtpserver.close()

def acquire(restart,stop):
    #file_path="C:\\Users\\BeARING\\Dropbox\\UCB Air Monitor\\Data\\Roof\\PAVLOVSKY\\";
    file_path=r"C:\Users\ucbcr\Dropbox\UCB Air Monitor\Data\Roof\current"
    directory=file_path+"\\"+str(datetime.datetime.now().year)
    wait="wait det:LYNX01 /acq"

    if not os.path.exists(directory):
        os.makedirs(directory)

    while( datetime.datetime.now()<stop_acq_time ):
        subprocess.call(wait)
        out_file="\""+directory+"\\"+str(datetime.datetime.now()).replace(" ","_").replace(".","-").replace(":","-")+".cnf\""
        cmd="movedata det:LYNX01 "+out_file+" /overwrite"
        #print(cmd)
        subprocess.call(cmd)
        er=subprocess.call(restart)
        if ( er )
            print("There were problems restarting the next acquisition")
            print("Trying to restart...")
            subprocess.call(stop)
            if(subprocess.call(restart)):
                print("Could not restart, exitting")
                msg="The acq tried to restart, but couldn't. Here's the error number "+str(er)
                send_mail(msg)
                break
            else:
                print("Restart successful")
                msg="Restarted acq after failing. Suggested quality check"
                send_mail(msg)
        if(datetime.datetime.now() > warn_acq_time):
            msg="The acq has been running for 3.25 days. The filter should be changed every 3 days. Please tend to this before the deadline."
            send_mail(msg)
    return

def log_filter_start(log_name):
    fmt="%Y-%m-%d %H:%M:%S"
    tim=times(log_name,fmt)
    tim.append(datetime.datetime.now())
    with open(log_name,'w') as log_fil:
        for el in tim:
            log_fil.write(el.strftime(fmt)+"\n")
    return;
    

def times(log_name,fmt):
    lst=[]
    with open(log_name) as log_fil:
        for line in log_fil:
            if(line=='\n'):
                continue
            lst.append(datetime.datetime.strptime(line[:-1],fmt))
    return lst

    
def main():
    #start the detector
    restart="startmca det:LYNX01 /realpreset=300"
    stop="stopmca det:LYNX01"
    er=subprocess.call(restart)
    if(0==er):
        print("Detector opened with no errors!")
        msg="Run start at "+str(datetime.datetime.now())
        msg=msg+"\nCheck me at  "+str(three_days_from_now)
        log_filter_start('filter_start_times.dat')
        print(msg) 
        send_mail(msg)
    else:
        msg="The detector could not be opened. The error number is:"+str(er)
        print(msg)
        subprocess.call(stop)
        send_mail(msg)
        return

    acquire(restart,stop)
    er=subprocess.call(stop)
    msg="Run end.... RESTART ME!"
    send_mail(msg)

if __name__ == "__main__":
    main()
