import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
import pickle
import datetime as dt
import redis
import sys
from numpy import abs, sqrt, cos, sin, pi
from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1

from solenoidBDistribution import Ball as Bp


# Model

class CloakAgent():
    def __init__(self):
        self.hostIP = '10.32.247.50'
        self.hostPort=6379
        # coil
        self.I = 1.0
        self.coilRadius = 1.5e-2
        self.N = 100
        conductorWidth = 4e-3
        # conductorWidth = self.coilRadius/100.0
        if self.N % 2 == 1:
            self.Z0 = (self.N//2) * conductorWidth
        else:
            self.Z0 = (self.N//2 - 0.5) * conductorWidth
        self.coilZs = nu.linspace(-self.Z0, self.Z0, self.N)
        # magnets
        self.k_phi = 1e4
        self.FMThickness = 1e-3
        self.Z_lO = self.Z0-self.coilRadius/2+self.FMThickness
        self.Z_uO = self.Z0+self.coilRadius/2+self.FMThickness
        self.Z_lI = self.Z0-self.coilRadius/2-self.FMThickness
        self.Z_uI = self.Z0+self.coilRadius/2-self.FMThickness
        # plots
        self.plotLeftBoundCoeff = 0.1
        self.plotRightBoundCoeff = 0.9
        self.plotLowerBoundCoeff = 0.0
        self.plotUpperBoundCoeff = 1.2
        # points to be measured
        self.points = 50
        self.los = nu.linspace(self.plotLeftBoundCoeff, self.plotRightBoundCoeff, self.points) * self.coilRadius
        self.zs = nu.linspace(self.plotLowerBoundCoeff, self.plotUpperBoundCoeff, self.points) * self.Z0


    def run(self):
        # get mode string
        modeString = sys.argv[1]
        assert modeString != None
        # run as master
        if modeString.lower() == 'master' or modeString.lower() == 'm':
            try:
                self.runAsMasterOnCluster()
            except KeyboardInterrupt as e:
                pass
            finally:
                master = redis.Redis(host=self.hostIP, port=self.hostPort)
                shouldTerminateWorkers = input('Should terminate all workers? [y/n]: ')
                if shouldTerminateWorkers.lower() == 'y':
                    print('Terminating remote workers ...')
                    master.set('terminateFlag', 'True')
                    while master.rpop('cookedQueue') != None:
                        pass
                else:
                    print('Cleaning queues ...')
                    # clean queues
                    while master.rpop('rawQueue') != None:
                        pass
                    while master.rpop('cookedQueue') != None:
                        pass
                print('Successfully shutdown master program, bye-bye!')
        # run as slave
        elif modeString.lower() == 'slave' or modeString.lower() == 's':
            if len(sys.argv) <= 2:
                self.runAsSlaveOnCluster()
            else:
                self.runAsSlaveOnCluster(workerAmount=int(sys.argv[2]))
        # plot saved B Distribution
        elif modeString.lower() == 'b':
            self.__plotBFieldDistribution()
        # run as single mode
        elif modeString.lower() == '1':
            self.runAsSingle()
        # run as single mode in parallel
        elif modeString.lower() == 'p':
            self.runAsSingleInParallel()


    def runAsMasterOnCluster(self):
        master = redis.Redis(host=self.hostIP, port=self.hostPort)
        print('Master node starts.')
        # clean queues
        while master.rpop('rawQueue') != None:
            pass
        while master.rpop('cookedQueue') != None:
            pass
        master.set('terminateFlag', 'False')
        print('Queues cleaned-up.')
        print('Start main calculation')
        _start = dt.datetime.now()
        # start main calculation
        # generate all points and push them to raw queue.
        for i, lo in enumerate(self.los):
            args = (lo, self.zs, self.coilRadius, self.coilZs, self.FMThickness, self.Z0, self.Z_lO, self.Z_uO, self.Z_lI, self.Z_uI, self.I, self.k_phi)
            master.lpush('rawQueue', pickle.dumps(args))
        _amount = len(self.los)
        print('All {} tasks distributed. Waiting for slaves ...'.format(_amount))
        # collect calculated bs: [lo, z, bp_lo, bp_z]
        bs = nu.zeros((len(self.los)*len(self.zs), 4))
        collectedAmount = 0
        while collectedAmount < _amount:
            popResult = master.brpop(['cookedQueue'], 3)
            if popResult == None:
                continue
            _, binaryBp = popResult
            bs[collectedAmount*len(self.los):collectedAmount*len(self.los)+len(self.zs), :] = pickle.loads(binaryBp)
            collectedAmount += 1
        _end = dt.datetime.now()
        print('All {} trajectories generated. (cost {:.3g} hours)'.format(_amount, (_end-_start).total_seconds()/3600.0))
        # sort before save
        bs = nu.sort(bs, axis=0)
        # save results
        with open('bs.pickle', 'wb') as file:
            pickle.dump(bs, file)
        # plot
        self.__plotBFieldDistribution()


    def runAsSlaveOnCluster(self, workerAmount=min(int(mp.cpu_count()*0.75), 50), rawQueue='rawQueue', cookedQueue='cookedQueue'):
        workerTank = []
        shouldStop = mp.Event()
        shouldStop.clear()
        slave = redis.Redis(host=self.hostIP, port=self.hostPort)
        print(f'Slave node starts with {workerAmount} workers.')
        for _ in range(workerAmount):
            worker = mp.Process(target=computeBFieldInCluster, args=(rawQueue, cookedQueue, self.hostIP, self.hostPort, shouldStop))
            worker.start()
        while True:
            x = input("Press 'q' to stop local workers: ")
            if x.lower() == 'q':
                shouldStop.set()
                break
            # check remote flag
            elif slave.get('terminateFlag') != None:
                if slave.get('terminateFlag').decode() == 'True':
                    break
        for worker in workerTank:
            worker.join()


    def runAsSingle(self):
        _amount = len(self.los) * len(self.zs)
        count = 0
        bs = nu.zeros((_amount, 4))
        for i, lo in enumerate(self.los):
            for j, z in enumerate(self.zs):
                bp = Bp(lo, z, self.coilRadius, self.coilZs, self.FMThickness, self.Z0, self.Z_lO, self.Z_uO, self.Z_lI, self.Z_uI, self.I, self.k_phi)
                bs[count, :] = [lo, z, bp[0], bp[1]]
                count += 1
        # save results
        with open('bs.pickle', 'wb') as file:
            pickle.dump(bs, file)
        # plot
        self.__plotBFieldDistribution()


    def runAsSingleInParallel(self):
        _amount = len(self.los) * len(self.zs)
        args = []
        for i, lo in enumerate(self.los):
            for j, z in enumerate(self.zs):
                args.append((lo, z, self.coilRadius, self.coilZs, self.FMThickness, self.Z0, self.Z_lO, self.Z_uO, self.Z_lI, self.Z_uI, self.I, self.k_phi))
        with mp.Pool(processes=mp.cpu_count()*3//4) as pool:
            bs = pool.starmap(Bp, args)
        bs = nu.array(bs)
        # save results
        with open('bs.pickle', 'wb') as file:
            pickle.dump(bs, file)
        # plot
        self.__plotBFieldDistribution()


    def __plotBFieldDistribution(self):
        with open('bs.pickle', 'rb') as file:
            bs = pickle.load(file)
        bs_lo = []
        bs_z = []
        for b in bs:
            lo, z, bp_lo, bp_z = b
            bs_lo.append(bp_lo)
            bs_z.append(bp_z)
        bs_lo = nu.array(bs_lo)
        bs_z = nu.array(bs_z)

        _los, _zs = nu.meshgrid(self.los, self.zs, indexing='ij')
        pl.contourf(_los/self.coilRadius, _zs/self.Z0, bs_lo.reshape(self.points, self.points), levels=50)
        pl.colorbar()
        pl.show()
        pl.contourf(_los/self.coilRadius, _zs/self.Z0, bs_z.reshape(self.points, self.points), levels=50)
        pl.colorbar()
        pl.show()
        pl.title(r'Coil $B$ Distribution ' + f'(N={self.N})', fontsize=24)
        pl.xlabel(r'Relative Radius Position $\rho$/coilRadius [-]', fontsize=22)
        pl.ylabel(r'Relative Z Position $z$/coilHeight [-]', fontsize=22)
        pl.tick_params(labelsize=16)
        pl.quiver(_los/self.coilRadius, _zs/self.Z0, bs_lo, bs_z, label=r'$B$ field')
        pl.show()


def Baxis(lo, zs, coilRadius, coilZs, FMThickness, Z0, Z_lO, Z_uO, Z_lI, Z_uI, I, k_phi):
    baxis = nu.zeros((len(zs), 4))
    baxis[:, 0] = lo
    baxis[:, 1] = zs.ravel()
    for i, z in enumerate(zs):
        bp = Bp(lo, z, coilRadius, coilZs, FMThickness, Z0, Z_lO, Z_uO, Z_lI, Z_uI, I, k_phi)
        baxis[i, 2] = bp[0]
        baxis[i, 3] = bp[1]
    return baxis



def computeBFieldInCluster(rawQueue, cookedQueue, hostIP, hostPort, shouldStop):
    slave = redis.Redis(host=hostIP, port=hostPort)
    while shouldStop.is_set() == False:
        # check if terminated by master
        terminateFlag = slave.get('terminateFlag')
        if terminateFlag != None and terminateFlag.decode() == 'True':
            return
        # continue calculation
        # http://y0m0r.hateblo.jp/entry/20130320/1363786926, timeout=3sec
        popResult = slave.brpop([rawQueue], 3)
        if popResult == None:
            continue
        _, binaryArgs = popResult
        args = pickle.loads(binaryArgs)
        baxis = Baxis(*args)
        binaryBaxis = pickle.dumps(baxis)
        slave.lpush(cookedQueue, binaryBaxis)


# Main

if __name__ == '__main__':
    mp.freeze_support()
    cloakAgent = CloakAgent()
    cloakAgent.run()
