import socket, struct, time
import numpy as np

class ATISensor:
    '''The class interface for an ATI Force/Torque sensor.
    This class contains all the functions necessary to communicate
    with an ATI Force/Torque sensor with a Net F/T interface
    using socket.
    '''    
    def __init__(self, ip='192.168.1.1'):
        self.ip = ip
        self.port = 49151
        self.sock = socket.socket()
        time.sleep(0.5) # wait for proper connection
        self.sock.connect((self.ip, self.port))
        self.READ_CALIBRATION_INFO = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                              0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.READ_FORCE = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        self.RESET_FORCE = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01])
        self.countsPerForce = 1000000
        self.countsPerTorque = 1000000
        self.scaleFactors_force = 15260      # ATI Nano25 of SusTech
        self.scaleFactors_torque = 92
        self.sock.send(self.RESET_FORCE)
    def get_measurement(self):
        self.sock.send(self.READ_FORCE)
        force_info = self.sock.recv(16)
        header, status, ForceX, ForceY, ForceZ, TorqueX, TorqueY, TorqueZ = struct.unpack('!2H6h', force_info)
        raw = np.array([ForceX, ForceY, ForceZ, TorqueX, TorqueY, TorqueZ])
        force_torque = np.concatenate([raw[:3] * self.scaleFactors_force/self.countsPerForce, 
                                 raw[3:] * self.scaleFactors_torque/self.countsPerTorque])
        return force_torque
    
    def reset(self):
        self.sock.send(self.RESET_FORCE)
        
    def close(self):
        self.sock.close()
