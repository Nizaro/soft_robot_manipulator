## Code modified from a template for PCAN protocol with RMDX motor
## Author of the original template Guillaume SAMAIN

## Needed Imports
from PCANBasic import *
import os
import sys
import time
import csv
import math
import copy

## Checks if PCANBasic.dll is available, if not, the program terminates
try:
    m_objPCANBasic = PCANBasic()
    m_DLLFound = True
except :
    print("Unable to find the library: PCANBasic.dll !")
    m_DLLFound = False

class Initialisation():

    # Defines
    #region

    # Sets the PCANHandle (Hardware Channel)
    PcanHandle = PCAN_USBBUS1

    # Sets the desired connection mode (CAN = false / CAN-FD = true)
    IsFD = False

    # Sets the bitrate for normal CAN devices
    Bitrate = PCAN_BAUD_1M

    # Sets the bitrate for CAN FD devices. 
    # Example - Bitrate Nom: 1Mbit/s Data: 2Mbit/s:
    #   "f_clock_mhz=20, nom_brp=5, nom_tseg1=2, nom_tseg2=1, nom_sjw=1, data_brp=2, data_tseg1=3, data_tseg2=1, data_sjw=1"
    BitrateFD = b'f_clock_mhz=20, nom_brp=5, nom_tseg1=2, nom_tseg2=1, nom_sjw=1, data_brp=2, data_tseg1=3, data_tseg2=1, data_sjw=1'    
    #endregion

    # Members
    #region

    # Shows if DLL was found
    m_DLLFound = False

    #endregion

    def __init__(self):
        """
        Create an object starts the programm
        """
        
        ## Initialization of the selected channel
        stsResult = m_objPCANBasic.Initialize(self.PcanHandle,self.Bitrate)

        if stsResult != PCAN_ERROR_OK:
            print("Can not initialize. Please check the defines in the code.")
            self.ShowStatus(stsResult)
            print("")
            self.getInput("Press <Enter> to quit...")
            return

        print("Successfully initialized.")
        self.getInput("Press <Enter> to write...")
        strinput = "y"
        while strinput == "y":
            self.clear()
            break


    def __del__(self):
        if self.m_DLLFound:
            self.m_objPCANBasic.Uninitialize(PCAN_NONEBUS)

    def getInput(self, msg="Press <Enter> to continue...", default=""):
        res = default
        if sys.version_info[0] >= 3:
            res = input(msg + " ")
        else:
            res = raw_input(msg + " ")
        if len(res) == 0:
            res = default
        return res

    # Help-Functions
    #region
    def clear(self):
        """
        Clears the console
        """
        if os.name=='nt':
            os.system('cls')
        else:
            os.system('clear')

    def ShowStatus(self,status):
        """
        Shows formatted status

        Parameters:
            status = Will be formatted
        """
        print("=========================================================================================")
        print(self.GetFormattedError(status))
        print("=========================================================================================")

    def GetFormattedError(self, error):
        """
        Help Function used to get an error as text

        Parameters:
            error = Error code to be translated

        Returns:
            A text with the translated error
        """
        ## Gets the text using the GetErrorText API function. If the function success, the translated error is returned.
        ## If it fails, a text describing the current error is returned.
        stsReturn = m_objPCANBasic.GetErrorText(error,0x09)
        if stsReturn[0] != PCAN_ERROR_OK:
            return "An error occurred. Error-code's text ({0:X}h) couldn't be retrieved".format(error)
        else:
            message = str(stsReturn[1])
            return message.replace("'","",2).replace("b","",1)
    
    #endregion



class Moteur():

    # Sets the PCANHandle (Hardware Channel)
    PcanHandle = PCAN_USBBUS1

    #Gain proportionnel du controleur
    #Kpv=200
    Kpv=200

    def __init__(self, ListeMoteurs,Trajectoire,Theta,Phi):
        time.sleep(3)
        for i in range(len(ListeMoteurs)):
            stsResult = self.RunMessage(ListeMoteurs[i][0])
            self.CheckMessages(stsResult)
        print("WAIT")
        time.sleep(5)


#-------------INIT---------------------------
        it = 0
        CourantMax = 0
        test = False
        t=0
        dt3=0
        t_old=0
        SpeedD = []
        PosD = []
        Angle = []
        AngleInit = []
        AngleZero=[]
        AngleRelativ=[]
        AngleAcquired= [False for i in range(len(ListeMoteurs))]
        AngleRelativ=[0 for i in range(len(ListeMoteurs))]
        Step=0
        Reach=0
        
        Angle_old=[]
        Courant = []
        Vit = []
        data = []
        Courant_Filtre=[]
        CourantD=[]
        err_old=[]
        err_int=[]
        maxdt = 0
        for i in range(len(ListeMoteurs)): #voir pour faire un thread OpenMP
            SpeedD.append(0)
            PosD.append(180)
            Angle.append(0)
            AngleInit.append(0)
            Angle_old.append(0)
            Courant.append(0)
            CourantD.append(-10)
            Vit.append(0)
            Courant_Filtre.append(0)
            err_old.append(0)
            err_int.append(0)
        temps = time.time()
        temps1 = temps
        old_temps = temps
        
#---------------------------------------------------------------



        while (True):
            #Mesure des angles
            for i in range(len(ListeMoteurs)): #voir pour faire un thread
                    
                #stsResult = self.CurrentMessage(ListeMoteurs[i], 10)
                stsResult = self.AngleMessage(ListeMoteurs[i][0])
                self.CheckMessages(stsResult)
            #Prise de zéro ou correction si déjà effectuée
            ListeData = self.ReadMessage()
            Data = self.ProcessData(ListeMoteurs, ListeData, Courant, Angle)
            Angle = Data[1]
            Courant = Data[0]
            if any(AngleAcquired[i]==False for i in range(len(ListeMoteurs))):
                if time.time()-temps1>15 :
                    print("unable to set zero") # Occurs when motor fail to reply with an angular position
                    return
                AngleZero = copy.deepcopy(Data[1])
                print("AngleZero=",AngleZero)
                AngleAcquired=copy.deepcopy(Data[3])
                if all(AngleAcquired[i]==True for i in range(len(ListeMoteurs))) :
                    print("Prise de zéro effectuée") # All motor angular position need to be set
            else:
                for i in range(len(ListeMoteurs)):
                    AngleRelativ[i]=Angle[i]-AngleZero[i]
            print("Angle relatif =",AngleRelativ)
            
            
            #Commande en position
            if all(AngleAcquired[i]==True for i in range(len(ListeMoteurs))):
                if Step==Reach:
                    Step+=1
                    #file= open("Tangential_Trajectory.txt","a")
                    #file.write("\n"+time.ctime(time.time())+";"+str(Theta[Step-1][0])+";"+str(Theta[Step-1][0])+";"+str(Phi[Step-1][0])+";"+str(Phi[Step-1][1])+";"+str(Trajectoire[Step-1][0])+";"+str(Trajectoire[Step-1][1])+";"+str(Trajectoire[Step-1][2])+";"+str(Trajectoire[Step-1][3]))
                    #file.close()
                    input("Press Enter to start Step "+str(Step)+" :"+str(Trajectoire[Step-1]))
                for i in range(len(ListeMoteurs)):
                    # Position command to motor
                    stsResult = self.PositionMessage(ListeMoteurs[i][0],int((Trajectoire[Step-1][i]+AngleZero[i])*100*ListeMoteurs[i][1]),int(abs(300*ListeMoteurs[i][1]**3/1500)))
                    self.CheckMessages(stsResult)
            if all(AngleRelativ[i]>Trajectoire[Step-1][i]-1 and AngleRelativ[i]<Trajectoire[Step-1][i]+1 for i in range(len(ListeMoteurs))):
                Reach=Step
                if Step==len(Trajectoire):
                    print("fin de la trajectoire")
                    return
            
            #Sécurité en courant
            if abs(Courant[0])>CourantMax:
                CourantMax=abs(Courant[0])
            if CourantMax>3:
                print("STOOOOOOOP")
                for i in range(len(ListeMoteurs)):
                    stsResult = self.OffMessage(ListeMoteurs[i][0])
                    self.CheckMessages(stsResult)

            temps = time.time()
            dt = temps-old_temps
            if dt > maxdt :
                 maxdt = dt
            while(temps-old_temps<0.3):
                temps=time.time()
            old_temps = temps



    def RunMessage(self, MoteurID):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0x88
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[3] = 0x00
        msgCanMessage.DATA[4] = 0x00
        msgCanMessage.DATA[5] = 0x00
        msgCanMessage.DATA[6] = 0x00
        msgCanMessage.DATA[7] = 0x00
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    def OffMessage(self, MoteurID):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0x80
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[3] = 0x00
        msgCanMessage.DATA[4] = 0x00
        msgCanMessage.DATA[5] = 0x00
        msgCanMessage.DATA[6] = 0x00
        msgCanMessage.DATA[7] = 0x00
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    def SpeedMessage(self, MoteurID, SpeedD):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0xA2
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[3] = 0x00
        msgCanMessage.DATA[4] = SpeedD
        msgCanMessage.DATA[5] = SpeedD >> 8
        msgCanMessage.DATA[6] = SpeedD >> 16
        msgCanMessage.DATA[7] = SpeedD >> 32
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    def PositionMessage(self, MoteurID, PositionD,MaxSpeed):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0xA3
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[2] = MaxSpeed
        msgCanMessage.DATA[3] = MaxSpeed >> 8
        msgCanMessage.DATA[4] = PositionD
        msgCanMessage.DATA[5] = PositionD >> 8
        msgCanMessage.DATA[6] = PositionD >> 16
        msgCanMessage.DATA[7] = PositionD >> 32
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    def CurrentMessage(self, MoteurID, CurrentD):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0xA1
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[3] = 0x00
        msgCanMessage.DATA[4] = CurrentD
        msgCanMessage.DATA[5] = CurrentD >> 8
        # msgCanMessage.DATA[4] = 0xE7
        # msgCanMessage.DATA[5] = 0xFF
        # msgCanMessage.DATA[4] = 0x19
        # msgCanMessage.DATA[5] = 0x00
        msgCanMessage.DATA[6] = 0x00
        msgCanMessage.DATA[7] = 0x00
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    
    def MotorStatusMessage(self, MoteurID):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0x9C
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[3] = 0x00
        msgCanMessage.DATA[4] = 0x00
        msgCanMessage.DATA[5] = 0x00
        msgCanMessage.DATA[6] = 0x00
        msgCanMessage.DATA[7] = 0x00
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    def AngleMessage(self, MoteurID):
        msgCanMessage = TPCANMsg()
        msgCanMessage.ID = MoteurID #0x141
        msgCanMessage.LEN = 8
        msgCanMessage.MSGTYPE = PCAN_MESSAGE_STANDARD.value
        msgCanMessage.DATA[0] = 0x92
        msgCanMessage.DATA[1] = 0x00
        msgCanMessage.DATA[3] = 0x00
        msgCanMessage.DATA[4] = 0x00
        msgCanMessage.DATA[5] = 0x00
        msgCanMessage.DATA[6] = 0x00
        msgCanMessage.DATA[7] = 0x00
        #print(msgCanMessage.DATA[0], msgCanMessage.DATA[1], msgCanMessage.DATA[2], msgCanMessage.DATA[3], msgCanMessage.DATA[4], msgCanMessage.DATA[5], msgCanMessage.DATA[6], msgCanMessage.DATA[7])
        return m_objPCANBasic.Write(self.PcanHandle, msgCanMessage)

    def CheckMessages(self, stsResult):

        ## Checks if the message was sent
        if (stsResult != PCAN_ERROR_OK):
            self.ShowStatus(stsResult)
        #else:
            #print("Message was successfully SENT")

    def ReadMessages(self):
        """
        Function for reading PCAN-Basic messages
        """
        stsResult = PCAN_ERROR_OK  # Pas d'erreur

        ## We read at least one time the queue looking for messages. If a message is found, we look again trying to
        ## find more. If the queue is empty or an error occurr, we get out from the dowhile statement.
        # PCAN_ERROR_QRCVEMPTY : Receive queue is empty
        while (not (stsResult & PCAN_ERROR_QRCVEMPTY)):
            if self.IsFD:
                stsResult = self.ReadMessageFD()
            else:
                stsResult = self.ReadMessage()
            if stsResult != PCAN_ERROR_OK and stsResult != PCAN_ERROR_QRCVEMPTY:
                self.ShowStatus(stsResult)
            return

    def ReadMessage(self):
        """
        Function for reading CAN messages on normal CAN devices

        Returns:
            A TPCANStatus error code
        """
        stsResult = PCAN_ERROR_OK  # Pas d'erreur
        ListeData=[]
        i=0

        while (not (stsResult & PCAN_ERROR_QRCVEMPTY)):
            ## We execute the "Read" function of the PCANBasic tant qu'il n'y a pas de pb et que la queue n'est pas vide
            #print("qekyrgkhqdfg")
            NewData = m_objPCANBasic.Read(self.PcanHandle) #Store dans DATA un code d'erreur, le message CAN et le temps
            ListeData.append(NewData)
            stsResult = ListeData[i][0]
            if stsResult != PCAN_ERROR_OK and stsResult != PCAN_ERROR_QRCVEMPTY:
                self.ShowStatus(stsResult)
            #print("ID: " + self.GetIdString(NewData[1].ID, NewData[1].MSGTYPE))
            #print("Data: " + self.GetDataString(NewData[1].DATA, NewData[1].MSGTYPE))
            i=i+1
        #print("Nb messages : " + str(len(ListeData)))
        return ListeData

    def ProcessData(self, ListeMoteurs, ListeData, Courant, Angle):
        Couple=[]
        VitReelle=[]
        AngleAcquired=[False for i in range(len(ListeMoteurs))]

        for j in range(len(ListeMoteurs)):
            Couple.append(0)
            VitReelle.append(0)
            i = 0

            while i < len(ListeData):

                if ListeMoteurs[j][0] == ListeData[i][1].ID:

                    if ListeData[i][0] == PCAN_ERROR_OK:

                        if ListeData[i][1].DATA[0] == 0xA1 or ListeData[i][1].DATA[0] == 0x9C:
                            Couple[j] = ListeData[i][1].DATA[3]*pow(256,1) + ListeData[i][1].DATA[2]
                            if Couple[j] > 3000:
                                Couple[j] = self.twosComplement_hex(Couple[j])
                            Courant[j] = (Couple[j]*33)/2048
                            VitReelle[j]= ListeData[i][1].DATA[5]*pow(256,1) + ListeData[i][1].DATA[4]
                            if VitReelle[j] > 30000:
                                VitReelle[j] = self.twosComplement_hex(VitReelle[j])
                            #print("fdgfswxvfv")
                        elif ListeData[i][1].DATA[0] == 0x92:
                            AngleAcquired[j]=True
                            Angle[j] = ListeData[i][1].DATA[6]*pow(256,5) + ListeData[i][1].DATA[5]*pow(256,4) + ListeData[i][1].DATA[4]*pow(256,3) + ListeData[i][1].DATA[3]*pow(256,2) + ListeData[i][1].DATA[2]*pow(256,1) + ListeData[i][1].DATA[1]
                            if ListeData[i][1].DATA[7] == 1:
                                Angle[j]=-Angle[j]
                            Angle[j]=Angle[j]/(100*ListeMoteurs[j][1])
                    ListeData.pop(i)
                    i = i-1
                i = i+1
                    
        return Courant, Angle, VitReelle, AngleAcquired

    def twosComplement_hex(self, val):
        bits = 16
        #val = int(hexval, bits)
        if val & (1 << (bits-1)):
            val -= 1 << bits
        return val  


    def ReadMessageFD(self):
        """
        Function for reading messages on FD devices

        Returns:
            A TPCANStatus error code
        """
        ## We execute the "Read" function of the PCANBasic    
        stsResult = m_objPCANBasic.ReadFD(self.PcanHandle)

        if stsResult[0] == PCAN_ERROR_OK:
            ## We show the received message
            self.ProcessMessageCanFd(stsResult[1],stsResult[2])
            
        return stsResult[0]

    def ShowStatus(self,status):
        """
        Shows formatted status

        Parameters:
            status = Will be formatted
        """
        print("=========================================================================================")
        print(self.GetFormattedError(status))
        print("=========================================================================================")

    def GetLengthFromDLC(dlc):
        """
        Gets the data length of a CAN message

        Parameters:
            dlc = Data length code of a CAN message

        Returns:
            Data length as integer represented by the given DLC code
        """
        if dlc == 9:
            return 12
        elif dlc == 10:
            return 16
        elif dlc == 11:
            return 20
        elif dlc == 12:
            return 24
        elif dlc == 13:
            return 32
        elif dlc == 14:
            return 48
        elif dlc == 15:
            return 64
        
        return dlc

    def GetIdString(self, id, msgtype):
        """
        Gets the string representation of the ID of a CAN message

        Parameters:
            id = Id to be parsed
            msgtype = Type flags of the message the Id belong

        Returns:
            Hexadecimal representation of the ID of a CAN message
        """
        if (msgtype & PCAN_MESSAGE_EXTENDED.value) == PCAN_MESSAGE_EXTENDED.value:
            return '%.8Xh' %id
        else:
            return '%.3Xh' %id

    def GetTimeString(self, time):
        """
        Gets the string representation of the timestamp of a CAN message, in milliseconds

        Parameters:
            time = Timestamp in microseconds

        Returns:
            String representing the timestamp in milliseconds
        """
        fTime = time / 1000.0
        return '%.1f' %fTime

    def GetTypeString(self, msgtype):  
        """
        Gets the string representation of the type of a CAN message

        Parameters:
            msgtype = Type of a CAN message

        Returns:
            The type of the CAN message as string
        """
        if (msgtype & PCAN_MESSAGE_STATUS.value) == PCAN_MESSAGE_STATUS.value:
            return 'STATUS'
        
        if (msgtype & PCAN_MESSAGE_ERRFRAME.value) == PCAN_MESSAGE_ERRFRAME.value:
            return 'ERROR'        
        
        if (msgtype & PCAN_MESSAGE_EXTENDED.value) == PCAN_MESSAGE_EXTENDED.value:
            strTemp = 'EXT'
        else:
            strTemp = 'STD'

        if (msgtype & PCAN_MESSAGE_RTR.value) == PCAN_MESSAGE_RTR.value:
            strTemp += '/RTR'
        else:
            if (msgtype > PCAN_MESSAGE_EXTENDED.value):
                strTemp += ' ['
                if (msgtype & PCAN_MESSAGE_FD.value) == PCAN_MESSAGE_FD.value:
                    strTemp += ' FD'
                if (msgtype & PCAN_MESSAGE_BRS.value) == PCAN_MESSAGE_BRS.value:                    
                    strTemp += ' BRS'
                if (msgtype & PCAN_MESSAGE_ESI.value) == PCAN_MESSAGE_ESI.value:
                    strTemp += ' ESI'
                strTemp += ' ]'
                
        return strTemp

    def GetDataString(self, data, msgtype):
        """
        Gets the data of a CAN message as a string

        Parameters:
            data = Array of bytes containing the data to parse
            msgtype = Type flags of the message the data belong

        Returns:
            A string with hexadecimal formatted data bytes of a CAN message
        """
        if (msgtype & PCAN_MESSAGE_RTR.value) == PCAN_MESSAGE_RTR.value:
            return "Remote Request"
        else:
            strTemp = b""
            for x in data:
                strTemp += b'%.2X ' % x
            return str(strTemp).replace("'","",2).replace("b","",1)

    def GetFormattedError(self, error):
        """
        Help Function used to get an error as text

        Parameters:
            error = Error code to be translated

        Returns:
            A text with the translated error
        """
        ## Gets the text using the GetErrorText API function. If the function success, the translated error is returned.
        ## If it fails, a text describing the current error is returned.
        stsReturn = m_objPCANBasic.GetErrorText(error,0x09)
        if stsReturn[0] != PCAN_ERROR_OK:
            return "An error occurred. Error-code's text ({0:X}h) couldn't be retrieved".format(error)
        else:
            message = str(stsReturn[1])
            return message.replace("'","",2).replace("b","",1)