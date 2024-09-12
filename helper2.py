#!/usr/bin/python
# coding=utf-8

#  Copyright 2024 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
#
# Plant Trait Estimation from Radiance for Fire Risk Estimation
# Author: Nimrod Carmon nimrod.carmon@jpl.nasa.gov

import os
import numpy as np
from numpy import pi
import pdb
from numba import njit
import numba as nb


def calcLidf(LIDFa: float, LIDFb: float):
    # the default values are for spherical distribytion
    #TypeLidf=1, LIDFa=-0.35,LIDFb=-0.15
    TypeLidf = 1
    #LIDFa = LIDF[0]
    #LIDFb = LIDF[1]

    if(TypeLidf==1):
        #pdb.set_trace()
        lidf= _dladgen(LIDFa,LIDFb)

    elif(TypeLidf==2):
        na=13
        lidf=_calc_LIDF_ellipsoidal(na,LIDFa)
    return lidf

@njit
def _dladgen(a,b):
    t=np.zeros(13)
    freq=np.zeros(13)
    for i in range(13):
        if i<=7:
            t[i]=(i+1)*10. 
            freq[i]=_dcum(a,b,t[i])
        elif i>=8 and i<12:
            t[i]=80.+(i-7)*2.
            freq[i]=_dcum(a,b,t[i])
        else:
            freq[i]=1.
    for i in range(12, 0, -1):
        if i>=1:
            freq[i]=freq[i]-freq[i-1]
    return freq

@njit
def _dcum(a,b,t):
    if (a>1.):
        dcum=1.-np.cos(np.radians(t))
    else:
        eps=1e-8
        delx=1.
        x=2*np.radians(t)
        p=x
        while (delx>eps):
            y = a*np.sin(x)+.5*b*np.sin(2.*x)
            dx=.5*(y-x+p)
            x=x+dx
            delx=np.absolute(dx)
        dcum=(2.*y+p)/pi
    return dcum

def _calc_LIDF_ellipsoidal(na,alpha):
    freq= _campbell(na,alpha)
    return freq


def _campbell(n,ala):
    """
    Computation of the leaf angle distribution function value (freq) 
    Ellipsoidal distribution function caracterised by the average leaf 
    inclination angle in degree (ala)                                  
    Campbell 1986                                                      
    """

    tx2=np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])
    tx1=np.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88., 90.])
    
    tl1=tx1*np.arctan(1.)/45.
    tl2=tx2*np.arctan(1.)/45.
    excent=np.exp(-1.6184e-5*ala**3+2.1145e-3*ala**2-1.2390e-1*ala+3.2491)
    x1  = excent/(np.sqrt(1.+excent**2*np.tan(tl1)**2))
    x2  = excent/(np.sqrt(1.+excent**2*np.tan(tl2)**2))
    if (excent==1.):
        freq = np.absolute(np.cos(tl1)-np.cos(tl2))
    else:
        alpha  = excent/np.sqrt(np.absolute(1.-excent**2))
        alpha2 = alpha**2
        x12 = x1**2
        x22 = x2**2
        if (excent>1):
            alpx1 = np.sqrt(alpha2+x12)
            alpx2 = np.sqrt(alpha2+x22)
            dum   = x1*alpx1+alpha2*np.log(x1+alpx1)
            freq  = np.absolute(dum-(x2*alpx2+alpha2*np.log(x2+alpx2)))
        else:
            almx1 = np.sqrt(alpha2-x12)
            almx2 = np.sqrt(alpha2-x22)
            dum   = x1*almx1+alpha2*np.arcsin(x1/alpha)
            freq  = np.absolute(dum-(x2*almx2+alpha2*np.arcsin(x2/alpha)))
    sum0 = np.sum(freq)
    freq=freq/sum0	#*100.

    return freq

# this is the slow version

def prospect_5B(N: float, Cab: float, Car: float, Cbrown: float, Cw: float, Cm: float, spectra):
    """PROSPECT model, by Jean-Baptiste Feret & Stephane Jacquemoud

    Féret J.B., François C., Asner G.P., Gitelson A.A., Martin R.E., Bidel L.P.R.,
    Ustin S.L., le Maire G., Jacquemoud S. (2008), PROSPECT-4 and 5: Advances in the
    leaf optical properties model separating photosynthetic pigments, Remote Sennp.sing
    of Environment, 112:3030-3043.
    Jacquemoud S., Ustin S.L., Verdebout J., Schmuck G., Andreoli G., Hosgood B.
    (1996), Estimating leaf biochemistry unp.sing the PROSPECT leaf optical properties
    model, Remote Sennp.sing of Environment, 56:194-202.
    Jacquemoud S., Baret F. (1990), PROSPECT: a model of leaf optical properties
    spectra, Remote Sensing of Environment, 34:75-91.
    """

    k = ( Cab*np.array(spectra[2]) + Car*np.array(spectra[3]) + Cbrown*np.array(spectra[4]) + Cw*np.array(spectra[5]) + Cm*np.array(spectra[6]) ) / N
    #import pdb; pdb.set_trace()
    refractive=np.array(spectra[1])
    # I think k is like the mean or normalized spctra given the trait scalars
    # ********************************************************************************
    # reflectance and transmittance of one layer
    # ********************************************************************************
    # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969), Interaction of
    # isotropic ligth with a compact plant leaf, Journal of the Optical Society of
    # American, 59:1376-1379.
    # ********************************************************************************
    
    # np.exponential integral: S13AAF routine from the NAG library

    # inputs: I guess leaf proprties??
    # output: two spectra - rho and tau
    tau=np.zeros(k.size)
    xx=np.zeros(k.size)
    yy=np.zeros(k.size)
    
    for i in range(tau.size):
        if k[i]<=0.0:
            tau[i]=1
        elif (k[i]>0.0 and k[i]<=4.0):
            xx[i]=0.5*k[i]-1.0
            # what the hell is this??
            # what are these numbers?? why not explained????
            yy[i]=(((((((((((((((-3.60311230482612224e-13 
                *xx[i]+3.46348526554087424e-12)*xx[i]-2.99627399604128973e-11) 
                *xx[i]+2.57747807106988589e-10)*xx[i]-2.09330568435488303e-9) 
                *xx[i]+1.59501329936987818e-8)*xx[i]-1.13717900285428895e-7) 
                *xx[i]+7.55292885309152956e-7)*xx[i]-4.64980751480619431e-6) 
                *xx[i]+2.63830365675408129e-5)*xx[i]-1.37089870978830576e-4) 
                *xx[i]+6.47686503728103400e-4)*xx[i]-2.76060141343627983e-3) 
                *xx[i]+1.05306034687449505e-2)*xx[i]-3.57191348753631956e-2) 
                *xx[i]+1.07774527938978692e-1)*xx[i]-2.96997075145080963e-1
            yy[i]=(yy[i]*xx[i]+8.64664716763387311e-1)*xx[i]+7.42047691268006429e-1
            yy[i]=yy[i]-np.log(k[i])
            tau[i]=(1.0-k[i])*np.exp(-k[i])+k[i]**2*yy[i]
        elif (k[i]>4.0 and k[i]<=85.0):
            xx[i]=14.5/(k[i]+3.25)-1.0
            yy[i]=(((((((((((((((-1.62806570868460749e-12 
                *xx[i]-8.95400579318284288e-13)*xx[i]-4.08352702838151578e-12) 
                *xx[i]-1.45132988248537498e-11)*xx[i]-8.35086918940757852e-11) 
                *xx[i]-2.13638678953766289e-10)*xx[i]-1.10302431467069770e-9) 
                *xx[i]-3.67128915633455484e-9)*xx[i]-1.66980544304104726e-8) 
                *xx[i]-6.11774386401295125e-8)*xx[i]-2.70306163610271497e-7) 
                *xx[i]-1.05565006992891261e-6)*xx[i]-4.72090467203711484e-6) 
                *xx[i]-1.95076375089955937e-5)*xx[i]-9.16450482931221453e-5) 
                *xx[i]-4.05892130452128677e-4)*xx[i]-2.14213055000334718e-3
            yy[i]=((yy[i]*xx[i]-1.06374875116569657e-2)*xx[i]-8.50699154984571871e-2)*xx[i]+9.23755307807784058e-1
            yy[i]=np.exp(-k[i])*yy[i]/k[i]
            tau[i]=(1.0-k[i])*np.exp(-k[i])+k[i]**2*yy[i]
        else:
            tau[i]=0
    
    # transmissivity of the layer
    
    theta1=90.
    t1= _tav_abs(theta1,refractive)
    theta2=40.
    t2= _tav_abs(theta2,refractive)
    x1=1-t1
    x2=t1**2*tau**2*(refractive**2-t1)
    x3=t1**2*tau*refractive**2
    x4=refractive**4-tau**2*(refractive**2-t1)**2
    x5=t2/t1
    x6=x5*(t1-1)+1-t2
    r=x1+x2/x4
    t=x3/x4
    ra=x5*r+x6
    ta=x5*t
    
    # ********************************************************************************
    # reflectance and transmittance of N layers
    # ********************************************************************************
    # Stokes G.G. (1862), On the intensity of the light reflected from or transmitted
    # through a pile of plates, Proceedings of the Royal Society of London, 11:545-556.
    # ********************************************************************************
    
    delta=(t**2-r**2-1)**2-4*r**2
    beta=(1+r**2-t**2-delta**0.5)/(2*r)
    va=(1+r**2-t**2+delta**0.5)/(2*r)
    vb=(beta*(va-r)/(va*(beta-r)))**0.5
    s1=ra*(va*vb**(N-1)-va**(-1)*vb**(-(N-1)))+(ta*t-ra*r)*(vb**(N-1)-vb**(-(N-1)))
    s2=ta*(va-va**(-1))
    s3=va*vb**(N-1)-va**(-1)*vb**(-(N-1))-r*(vb**(N-1)-vb**(-(N-1)))
    RN=s1/s3
    TN=s2/s3
    return RN, TN



@nb.njit
def _tav_abs(theta, refr):
    """Computation of the average transmittivity at the leaf surface within a given
    solid angle. teta is the incidence solid angle (in radian). The average angle
    that works in most cases is 40deg*pi/180. ref is the refaction index.

    Stern F. (1964), Transmission of isotropic radiation across an interface between
    two dielectrics, Applied Optics, 3:111-113.
    Allen W.A. (1973), Transmission of isotropic light across a dielectric surface in
    two and three dimensions, Journal of the Optical Society of America, 63:664-666.
    """
    thetarad = theta * np.pi / 180.0 #deg to radians

    res = np.zeros_like(refr)
    
    for i in range(refr.size):

        if theta == 0.:
            res[i] = 4. * refr[i] / (refr[i] + 1.) ** 2

        else:
            refr2 = refr[i] * refr[i]
            ax = (refr[i] + 1.) ** 2 / 2.
            bx = -(refr2 - 1.) ** 2 / 4.
            
            aa = np.sin(thetarad) ** 2
            
            if thetarad == np.pi / 2.:
                b1 = 0.
            else:
                b1 = np.sqrt((aa - (refr2 + 1.) / 2.) ** 2 + bx)

            b2 = aa - (refr2 + 1.) / 2.
            b0 = b1 - b2
            ts = (bx**2 / (6. * b0 ** 3) + bx / b0 - b0 / 2.) - (bx ** 2 / (6. * ax ** 3) + bx / ax - ax / 2.)
            tp2 = 0.0
            tp4 = 0.0
            tp1 = -2. * refr2 * (b0 - ax) / (refr2 + 1.)**2
            tp2 = -2. * refr2 * (refr2 + 1.) * np.log(b0 / ax) / (refr2 - 1.)**2
            tp3 = refr2 * (1. / b0 - 1. / ax) / 2.
            tp4 = 16. * refr2**2 * (refr2**2 + 1.) * np.log((2. * (refr2 + 1.) * b0 - (refr2 - 1.)**2) /
                                                            (2. * (refr2 + 1.) * ax - (refr2 - 1.)**2)) / (
                                                        (refr2 + 1.)**3 * (refr2 - 1.)**2)
            tp5 = 16. * refr2**3 * (1. / (2. * (refr2 + 1.) * b0 - ((refr2 - 1.)**2)) - 1. / (2. * (refr2 + 1.) *
                                                                                            ax - (refr2 - 1.)**2)) / (
                                                          refr2 + 1.)**3
            tp = tp1 + tp2 + tp3 + tp4 + tp5
            res[i] = (ts + tp) / (2. * np.sin(thetarad)**2)

    return res



@njit
def _tav_abs3(theta,refr):
        """Computation of the average transmittivity at the leaf surface within a given
        solid angle. teta is the incidence solid angle (in radian). The average angle
        that works in most cases is 40deg*pi/180. ref is the refaction index.

        Stern F. (1964), Transmission of isotropic radiation across an interface between
        two dielectrics, Applied Optics, 3:111-113.
        Allen W.A. (1973), Transmission of isotropic light across a dielectric surface in
        two and three dimensions, Journal of the Optical Society of America, 63:664-666.
        """
        #pdb.set_trace()
        #refr=np.array(refr)
        thetarad=np.radians(theta)
        res=np.zeros(refr.size)
        if (theta == 0.):
            res=4.*refr/(refr+1.)**2
        else:
            refr2=refr*refr
            ax=(refr+1.)**2/2.
            bx=-(refr2-1.)**2/4.
            
            if (thetarad == pi/2.):
                b1=0.
            else:
                b1=((np.sin(thetarad)**2-(refr2+1.)/2.)**2+bx)**0.5
            #pdb.set_trace()

            aa = np.ones(refr2.shape, dtype=np.float64) * np.sin(thetarad)**2
            #aa = [np.sin(thetarad)**2] * len(refr2)
            bb = (refr2+1.)/2.
            b2 = aa - bb
            #b2=np.sin(thetarad)**2-(refr2+1.)/2.

            b0=b1-b2
            ts=(bx**2/(6.*b0**3)+bx/b0-b0/2.)-(bx**2/(6.*ax**3)+bx/ax-ax/2.)
            tp2=np.zeros(refr.size)
            tp4=np.zeros(refr.size)
            tp1=-2.*refr2*(b0-ax)/(refr2+1.)**2
            tp2=-2.*refr2*(refr2+1.)*np.log(b0/ax)/(refr2-1.)**2
            tp3=refr2*(1./b0-1./ax)/2.
            tp4=16.*refr2**2*(refr2**2+1.)*np.log((2.*(refr2+1.)*b0-(refr2-1.)**2)/ 
                    (2.*(refr2+1.)*ax-(refr2-1.)**2))/((refr2+1.)**3*(refr2-1.)**2)
            tp5=16.*refr2**3*(1./(2.*(refr2+1.)*b0-((refr2-1.)**2))-1./(2.*(refr2+1.) 
                *ax-(refr2-1.)**2))/(refr2+1.)**3
            tp=tp1+tp2+tp3+tp4+tp5
            res=(ts+tp)/(2.*np.sin(thetarad)**2)
        
        return res



@njit
def _tav_abs2(theta,refr):
        """Computation of the average transmittivity at the leaf surface within a given
        solid angle. teta is the incidence solid angle (in radian). The average angle
        that works in most cases is 40deg*pi/180. ref is the refaction index.

        Stern F. (1964), Transmission of isotropic radiation across an interface between
        two dielectrics, Applied Optics, 3:111-113.
        Allen W.A. (1973), Transmission of isotropic light across a dielectric surface in
        two and three dimensions, Journal of the Optical Society of America, 63:664-666.
        """
        #pdb.set_trace()
        #refr=np.array(refr)
        thetarad=np.radians(theta)
        res=np.zeros(refr.size)
        if (theta == 0.):
            res=4.*refr/(refr+1.)**2
        else:
            refr2=refr*refr
            ax=(refr+1.)**2/2.
            bx=-(refr2-1.)**2/4.
            
            if (thetarad == pi/2.):
                b1=0.
            else:
                b1=((np.sin(thetarad)**2-(refr2+1.)/2.)**2+bx)**0.5
            
            b2=np.sin(thetarad)**2-(refr2+1.)/2.
            b0=b1-b2
            ts=(bx**2/(6.*b0**3)+bx/b0-b0/2.)-(bx**2/(6.*ax**3)+bx/ax-ax/2.)
            tp2=np.zeros(refr.size)
            tp4=np.zeros(refr.size)
            tp1=-2.*refr2*(b0-ax)/(refr2+1.)**2
            tp2=-2.*refr2*(refr2+1.)*np.log(b0/ax)/(refr2-1.)**2
            tp3=refr2*(1./b0-1./ax)/2.
            tp4=16.*refr2**2*(refr2**2+1.)*np.log((2.*(refr2+1.)*b0-(refr2-1.)**2)/ 
                    (2.*(refr2+1.)*ax-(refr2-1.)**2))/((refr2+1.)**3*(refr2-1.)**2)
            tp5=16.*refr2**3*(1./(2.*(refr2+1.)*b0-((refr2-1.)**2))-1./(2.*(refr2+1.) 
                *ax-(refr2-1.)**2))/(refr2+1.)**3
            tp=tp1+tp2+tp3+tp4+tp5
            res=(ts+tp)/(2.*np.sin(thetarad)**2)
        
        return res


def PRO4SAIL(rho,tau,lidf,lai,q,tts,tto,psi,rsoil):
    """
    This version has been implemented by Jean-Baptiste Féret
    Jean-Baptiste Féret takes the entire responsibility for this version 
    All comments, changes or questions should be sent to:
    jbferet@stanford.edu

    Jean-Baptiste Féret
    Institut de Physique du Globe de Paris
    Space and Planetary Geophysics
    October 2009
    this model PRO4SAIL is based on a version provided by
    Wout Verhoef 
    NLR 
    April/May 2003,
    original version downloadable at http://teledetection.ipgp.jussieu.fr/prosail/
    Improved and extended version of SAILH model that avoids numerical singularities
    and works more efficiently if only few parameters change.
    ferences:
    Verhoef et al. (2007) Unified Optical-Thermal Four-Stream Radiative
    Transfer Theory for Homogeneous Vegetation Canopies, IEEE TRANSACTIONS 
    ON GEOSCIENCE AND REMOTE SENSING, VOL. 45, NO. 6, JUNE 2007

    Args:
        rho (float): leaf reflectance
        tau (float): leaf transmmitance
        lidf (float): leaf distribution
        lai (float): leaf area index
        q (float): not sure
        tts (flat): solar zenith
        tto (float): observation zenith
        psi (float): I think it's a hot spot parameter
        rsoil (float): soil reflectance already mixed
    
    Returns:
        rsot (float): bi-directional reflectance factor
        rdot (float): hemispherical-directional reflectance factor in viewing direction
        rsdt (float): directional-hemispherical reflectance factor for solar incident flux
        rddt (float): bi-hemispherical reflectance factor
        
    """

    litab=np.array([5.,15.,25.,35.,45.,55.,65.,75.,81.,83.,85.,87.,89.])
    
    rd=pi/180.
    
    #	Geometric quantities
    cts		= np.cos(rd*tts)
    cto		= np.cos(rd*tto)
    ctscto	= cts*cto
    tants	= np.tan(rd*tts)
    tanto	= np.tan(rd*tto)
    cospsi	= np.cos(rd*psi)
    dso		= np.sqrt(tants*tants+tanto*tanto-2.*tants*tanto*cospsi)

    na=13

    

    # angular distance, compensation of shadow length
    #	Calculate geometric factors associated with extinction and scattering 
    #	Initialise sums

    ks	= 0
    ko	= 0
    bf	= 0
    sob	= 0
    sof	= 0

    #	Weighted sums over LIDF
    
    ttl = litab# leaf inclination discrete values
    ctl = np.cos(rd*ttl)
    #	SAIL volume scattering phase function gives interception and portions to be 
    #	multiplied by rho and tau
    for i in range(na):
        chi_s,chi_o,frho,ftau= _volscatt(tts,tto,psi,ttl[i])

    #********************************************************************************
    #*                   SUITS SYSTEM COEFFICIENTS 
    #*
    #*	ks  : Extinction coefficient for direct solar flux
    #*	ko  : Extinction coefficient for direct observed flux
    #*	att : Attenuation coefficient for diffuse flux
    #*	sigb : Backscattering coefficient of the diffuse downward flux
    #*	sigf : Forwardscattering coefficient of the diffuse upward flux
    #*	sf  : Scattering coefficient of the direct solar flux for downward diffuse flux
    #*	sb  : Scattering coefficient of the direct solar flux for upward diffuse flux
    #*	vf   : Scattering coefficient of upward diffuse flux in the observed direction
    #*	vb   : Scattering coefficient of downward diffuse flux in the observed direction
    #*	w   : Bidirectional scattering coefficient
    #********************************************************************************

        #	Extinction coefficients
        ksli = chi_s/cts
        koli = chi_o/cto

        #	Area scattering coefficient fractions
        sobli	= frho*pi/ctscto
        sofli	= ftau*pi/ctscto
        bfli	= ctl[i]*ctl[i]
        ks	= ks+ksli*lidf[i]
        ko	= ko+koli*lidf[i]
        bf	= bf+bfli*lidf[i]
        sob	= sob+sobli*lidf[i]
        sof	= sof+sofli*lidf[i]

    #	Geometric factors to be used later with rho and tau
    sdb	= 0.5*(ks+bf)
    sdf	= 0.5*(ks-bf)
    dob	= 0.5*(ko+bf)
    dof	= 0.5*(ko-bf)
    ddb	= 0.5*(1.+bf)
    ddf	= 0.5*(1.-bf)
    
    #if(flag[4]):
    #	Here rho and tau come in
    sigb= ddb*rho+ddf*tau
    sigf= ddf*rho+ddb*tau
    att	= 1.-sigf
    m2=(att+sigb)*(att-sigb)
    m2[np.where(m2<0)]=0
    m=np.sqrt(m2)
    sb	= sdb*rho+sdf*tau
    sf	= sdf*rho+sdb*tau
    vb	= dob*rho+dof*tau
    vf	= dof*rho+dob*tau
    w	= sob*rho+sof*tau
    
    #if (flag[5]):
    #	Here the LAI comes in
    #   Outputs for the case LAI = 0
    if (lai<=0):
        tss		= 1.
        too		= 1.
        tsstoo	= 1.
        rdd		= 0.
        tdd		= 1.
        rsd		= 0.
        tsd		= 0.
        rdo		= 0.
        tdo		= 0.
        rso		= 0.
        rsos	= 0.
        rsod	= 0.

        rddt	= rsoil
        rsdt	= rsoil
        rdot	= rsoil
        rsodt	= 0.
        rsost	= rsoil
        rsot	= rsoil
    
    else:
        #	Other cases (LAI > 0)
        e1		= np.exp(-m*lai)
        e2		= e1*e1
        rinf	= (att-m)/sigb
        rinf2	= rinf*rinf
        re		= rinf*e1
        denom	= 1.-rinf2*e2
    
        J1ks= _Jfunc1(ks,m,lai)
        J2ks= _Jfunc2(ks,m,lai)
        J1ko= _Jfunc1(ko,m,lai)
        J2ko= _Jfunc2(ko,m,lai)
    
        Ps = (sf+sb*rinf)*J1ks
        Qs = (sf*rinf+sb)*J2ks
        Pv = (vf+vb*rinf)*J1ko
        Qv = (vf*rinf+vb)*J2ko
    
        rdd	= rinf*(1.-e2)/denom
        tdd	= (1.-rinf2)*e1/denom
        tsd	= (Ps-re*Qs)/denom
        rsd	= (Qs-re*Ps)/denom
        tdo	= (Pv-re*Qv)/denom
        rdo	= (Qv-re*Pv)/denom
    
        tss	= np.exp(-ks*lai)
        too	= np.exp(-ko*lai)
        z	= _Jfunc3(ks,ko,lai)
        g1	= (z-J1ks*too)/(ko+m)
        g2	= (z-J1ko*tss)/(ks+m)
    
        Tv1 = (vf*rinf+vb)*g1
        Tv2 = (vf+vb*rinf)*g2
        T1	= Tv1*(sf+sb*rinf)
        T2	= Tv2*(sf*rinf+sb)
        T3	= (rdo*Qs+tdo*Ps)*rinf
    
        #	Multiple scattering contribution to bidirectional canopy reflectance
        rsod = (T1+T2-T3)/(1.-rinf2)
        
        #if (flag[6]):
        #	Treatment of the hotspot-effect
        alf=1e6
        #	Apply correction 2/(K+k) suggested by F.-M. Bréon
        if (q>0.):
            alf=(dso/q)*2./(ks+ko)
        if (alf>200.):	#inserted H. Bach 1/3/04
            alf=200.
        if (alf==0.):
            #	The pure hotspot - no shadow
            tsstoo = tss
            sumint = (1-tss)/(ks*lai)
        else:
            #	Outside the hotspot
            fhot=lai*np.sqrt(ko*ks)
            #	Integrate by exponential Simpson method in 20 steps
            #	the steps are arranged according to equal partitioning
            #	of the slope of the joint probability function
            x1=0.
            y1=0.
            f1=1.
            fint=(1.-np.exp(-alf))*.05
            sumint=0.
    
            for i in range(20):
                if (i<19):
                    x2=-np.log(1.-(i+1)*fint)/alf
                else:
                    x2=1.
                y2=-(ko+ks)*lai*x2+fhot*(1.-np.exp(-alf*x2))/alf 
                f2=np.exp(y2)
                sumint=sumint+(f2-f1)*(x2-x1)/(y2-y1)
                x1=x2
                y1=y2
                f1=f2
            tsstoo=f1
    
    #	Bidirectional reflectance
    #	Single scattering contribution
        rsos = w*lai*sumint
        
        #	Total canopy contribution
        rso=rsos+rsod
        
        #	Interaction with the soil
        dn=1.-rsoil*rdd
        
        # rddt: bi-hemispherical reflectance factor
        rddt=rdd+tdd*rsoil*tdd/dn
        # rsdt: directional-hemispherical reflectance factor for solar incident flux
        rsdt=rsd+(tsd+tss)*rsoil*tdd/dn
        # rdot: hemispherical-directional reflectance factor in viewing direction    
        rdot=rdo+tdd*rsoil*(tdo+too)/dn
        # rsot: bi-directional reflectance factor
        rsodt=rsod+((tss+tsd)*tdo+(tsd+tss*rsoil*rdd)*too)*rsoil/dn
        rsost=rsos+tsstoo*rsoil
        rsot=rsost+rsodt
    
    return rsot, rdot, rsdt, rddt


def _volscatt(tts,tto,psi,ttl):
    """
    tts     = solar zenith
    tto     = viewing zenith
    psi     = azimuth
    ttl     = leaf inclination angle
    chi_s   = interception functions
    chi_o   = interception functions
    frho    = function to be multiplied by leaf reflectance rho
    ftau    = functions to be multiplied by leaf transmittance tau
    """
    #	Compute volume scattering functions and interception coefficients
    #	for given solar zenith, viewing zenith, azimuth and leaf inclination angle.
    
    #	chi_s and chi_o are the interception functions.
    #	frho and ftau are the functions to be multiplied by leaf reflectance rho and
    #	leaf transmittance tau, respectively, in order to obtain the volume scattering
    #	function.
    
    #	Wout Verhoef, april 2001, for CROMA
    rd=pi/180.
    costs=np.cos(rd*tts)
    costo=np.cos(rd*tto)
    sints=np.sin(rd*tts)
    sinto=np.sin(rd*tto)
    cospsi=np.cos(rd*psi)
    
    psir=rd*psi
    
    costl=np.cos(rd*ttl)
    sintl=np.sin(rd*ttl)
    cs=costl*costs
    co=costl*costo
    ss=sintl*sints
    so=sintl*sinto
    
    #c ..............................................................................
    #c     betas -bts- and betao -bto- computation
    #c     Transition angles (beta) for solar (betas) and view (betao) directions
    #c     if thetav+thetal>pi/2, bottom side of the leaves is observed for leaf azimut 
    #c     interval betao+phi<leaf azimut<2pi-betao+phi.
    #c     if thetav+thetal<pi/2, top side of the leaves is always observed, betao=pi
    #c     same consideration for solar direction to compute betas
    #c ..............................................................................

    cosbts=5.
    if (np.absolute(ss)>1e-6):
        cosbts=-cs/ss
    
    cosbto=5.
    if (np.absolute(so)>1e-6):
        cosbto=-co/so
    
    if (np.absolute(cosbts)<1.):
        bts=np.arccos(cosbts)
        ds=ss
    else:
        bts=pi
        ds=cs
    
    chi_s=2./pi*((bts-pi*.5)*cs+np.sin(bts)*ss)
    
    if (np.absolute(cosbto)<1.):
        bto=np.arccos(cosbto)
        doo=so
    elif(tto<90.):
        bto=pi
        doo=co
    else:
        bto=0
        doo=-co
    
    chi_o=2./pi*((bto-pi*.5)*co+np.sin(bto)*so)
    
    #c ..............................................................................
    #c   Computation of auxiliary azimut angles bt1, bt2, bt3 used          
    #c   for the computation of the bidirectional scattering coefficient w              
    #c .............................................................................

    btran1=np.absolute(bts-bto)
    btran2=pi-np.absolute(bts+bto-pi)
    
    if (psir<=btran1):
        bt1=psir
        bt2=btran1
        bt3=btran2
    else:
        bt1=btran1
        if (psir<=btran2):
            bt2=psir
            bt3=btran2
        else:
            bt2=btran2
            bt3=psir
    
    t1=2.*cs*co+ss*so*cospsi
    t2=0.
    if (bt2>0.):
        t2=np.sin(bt2)*(2.*ds*doo+ss*so*np.cos(bt1)*np.cos(bt3))
    
    denom=2.*pi*pi
    frho=((pi-bt2)*t1+t2)/denom
    ftau=    (-bt2*t1+t2)/denom
    
    if (frho<0):
        frho=0
    
    if (ftau<0):
        ftau=0
    
    return chi_s,chi_o,frho,ftau

## J functions
def _Jfunc1(k, l, t):
    # Calculate d=(k-l)*t
    d = (k - l) * t

    # Initialize Jout with zeros
    Jout = np.zeros_like(d)

    # Calculate the absolute value of d outside the loop
    abs_d = np.absolute(d)

    # Calculate the indices where abs_d is greater than 1e-3
    valid_indices = np.where(abs_d > 1e-3)

    # Calculate the indices where abs_d is less than or equal to 1e-3
    invalid_indices = np.where(abs_d <= 1e-3)

    # Calculate Jout for valid indices
    Jout[valid_indices] = (np.exp(-l[valid_indices] * t) - np.exp(-k * t)) / (k - l[valid_indices])

    # Calculate Jout for invalid indices
    Jout[invalid_indices] = 0.5 * t * (np.exp(-k * t) + np.exp(-l[invalid_indices] * t)) * (1. - d[invalid_indices] * d[invalid_indices] / 12.)

    return Jout

def _Jfunc1_slow(k,l,t):
#	J1 function with avoidance of singularity problem
#	
    d=(k-l)*t
    Jout=np.zeros(d.size)
    for i in range(Jout.size):
        if (np.absolute(d[i])>1e-3):
            Jout[i]=(np.exp(-l[i]*t)-np.exp(-k*t))/(k-l[i])
        else:
            Jout[i]=0.5*t*(np.exp(-k*t)+np.exp(-l[i]*t))*(1.-d[i]*d[i]/12.)
    return Jout

def _Jfunc2(k,l,t):
    Jout=(1.-np.exp(-(k+l)*t))/(k+l)
    return Jout

def _Jfunc3(k,l,t):
#	J2 function
    Jout=(1.-np.exp(-(k+l)*t))/(k+l)
    return Jout


def canref(rsot, rdot, rsdt, rddt, E_dir, E_dif, tts):
    # Nimrod on Nov 2 2023.
    # It looks like what this function does is to account for the variability in direct\diffuse illumination onto the canopy
    # So they calculate the skyl parameter which I think is realted to the cos(i) parameter (or is 1-cos(i)), given the 
    # solar zenith tts. Then they scale the direct and diffuse fluxes acconrdingly.
    # Then because prosail gives different reflectance values as listed below, the scale and add these for the diffuse and direct
    # reflectances and divide by the total flux.
    # I think because of that it's ok to do but just make sure tts is correct.

    #    direct / diffuse light	#
    #
    # the direct and diffuse light are taken into account as proposed by:
    # Francois et al. (2002) Conversion of 400–1100 nm vegetation albedo 
    # measurements into total shortwave broadband albedo using a canopy 
    # radiative transfer model, Agronomie

    # rddt: bi-hemispherical reflectance factor
    # rsdt: directional-hemispherical reflectance factor for solar incident flux
    # rdot: hemispherical-directional reflectance factor in viewing direction    
    # rsot: bi-directional reflectance factor

    # ``tts`` -- Solar zenith angle (degrees)
    # ``tto`` -- View zenith angle (degrees)
    
    skyl2 = np.cos(np.deg2rad(tts))
    skyl = 0.847 - 1.61 * np.sin(np.radians(90 - tts)) + \
        1.04 * np.sin(np.radians(90 - tts)) * np.sin(np.radians(90 - tts)) # % diffuse radiation
    
    # Es = direct
    # Ed = diffuse
    # PAR direct
    #PARdiro	= (1-skyl)*E_dir
    
    dir_flux = (1 - skyl) * E_dir
    # PAR diffus
    #PARdifo	=	(skyl)*E_dif
    dif_flux = skyl * E_dif

    total_flux = dir_flux + dif_flux
    # resh : hemispherical reflectance
    resh = (rddt * dif_flux + rsdt * dir_flux) / total_flux
    # resv : directional reflectance
    resv = (rdot * dif_flux + rsot * dir_flux) / total_flux
    return resh, resv

def dataSpec_P5B(): # I've reorganized the data so it's separated by row in a CSV file - much easier for handling in Python.
    try:
       
        infile=open('/scratch/carmon/prosail/ProSAIL'+'/dataSpec_P5_resampled.csv', 'r')
        #infile=open(os.path.abspath(os.curdir)+'/dataSpec_P5_resampled.csv', 'r')
    except:
        print('Cannot open dataSpec_P5.csv, exiting.')
        headers, data= False, False
        return headers, data
    lines=infile.readlines()
    headers=[] # Row headers, describe what these data sets mean.  Not sure they're of much utility in the code.
    data=[]
    for line in lines:
        line=line.rstrip()
        vals=line.split(',')
        headers.append(vals[0])
        datavals=vals[1:]
        #if vals[0] == 'Wavelength (nm)':
        #    datavals=[int(v) for v in datavals]
        #else:
        datavals=[float(v) for v in datavals]
        data.append(datavals)
    infile.close()
    return headers, data


def prospect_5B2(N: float, Cab: float, Car: float, Cbrown: float, Cw: float, Cm: float, spectra):

    k = (Cab * np.array(spectra[2]) + Car * np.array(spectra[3]) + Cbrown * np.array(spectra[4]) +
         Cw * np.array(spectra[5]) + Cm * np.array(spectra[6])) / N
    
    refractive = np.array(spectra[1])
    refractive = refractive.astype(np.float64)

    tau = np.zeros(k.size)
    xx = np.zeros(k.size)
    yy = np.zeros(k.size)

    # Calculate tau using vectorized operations

    tau[k <= 0.0] = 1

    idx_1 = (k > 0.0) & (k <= 4.0)

    xx[idx_1] = 0.5 * k[idx_1] - 1.0

    yy[idx_1] = (((((((((((((((-3.60311230482612224e-13 * xx[idx_1] + 3.46348526554087424e-12) * xx[idx_1]
                         - 2.99627399604128973e-11) * xx[idx_1] + 2.57747807106988589e-10) * xx[idx_1]
                        - 2.09330568435488303e-9) * xx[idx_1] + 1.59501329936987818e-8) * xx[idx_1]
                       - 1.13717900285428895e-7) * xx[idx_1] + 7.55292885309152956e-7) * xx[idx_1]
                      - 4.64980751480619431e-6) * xx[idx_1] + 2.63830365675408129e-5) * xx[idx_1]
                     - 1.37089870978830576e-4) * xx[idx_1] + 6.47686503728103400e-4) * xx[idx_1]
                    - 2.76060141343627983e-3) * xx[idx_1] + 1.05306034687449505e-2) * xx[idx_1]
                   - 3.57191348753631956e-2) * xx[idx_1] + 1.07774527938978692e-1) * xx[idx_1] - 2.96997075145080963e-1

    yy[idx_1] = (yy[idx_1] * xx[idx_1] + 8.64664716763387311e-1) * xx[idx_1] + 7.42047691268006429e-1
    yy[idx_1] = yy[idx_1] - np.log(k[idx_1])
    tau[idx_1] = (1.0 - k[idx_1]) * np.exp(-k[idx_1]) + k[idx_1]**2 * yy[idx_1]

    idx_2 = (k > 4.0) & (k <= 85.0)
    xx[idx_2] = 14.5 / (k[idx_2] + 3.25) - 1.0

    yy[idx_2] = (((((((((((((((-1.62806570868460749e-12 * xx[idx_2] - 8.95400579318284288e-13) * xx[idx_2]
                         - 4.08352702838151578e-12) * xx[idx_2] - 1.45132988248537498e-11) * xx[idx_2]
                        - 8.35086918940757852e-11) * xx[idx_2] - 2.13638678953766289e-10) * xx[idx_2]
                       - 1.10302431467069770e-9) * xx[idx_2] - 3.67128915633455484e-9) * xx[idx_2]
                      - 1.66980544304104726e-8) * xx[idx_2] - 6.11774386401295125e-8) * xx[idx_2]
                     - 2.70306163610271497e-7) * xx[idx_2] - 1.05565006992891261e-6) * xx[idx_2]
                    - 4.72090467203711484e-6) * xx[idx_2] - 1.95076375089955937e-5) * xx[idx_2]
                   - 9.16450482931221453e-5) * xx[idx_2] - 4.05892130452128677e-4) * xx[idx_2] - 2.14213055000334718e-3

    yy[idx_2] = ((yy[idx_2] * xx[idx_2] - 1.06374875116569657e-2) * xx[idx_2] - 8.50699154984571871e-2) * xx[idx_2] + 9.23755307807784058e-1
    yy[idx_2] = np.exp(-k[idx_2]) * yy[idx_2] / k[idx_2]
    tau[idx_2] = (1.0 - k[idx_2]) * np.exp(-k[idx_2]) + k[idx_2]**2 * yy[idx_2]

    tau[k > 85.0] = 0

    # Transmissivity of the layer
    # we first calculate leaf transmmitance for two solid angles
    theta1 = 90. # there might be an incosistancy here with the paper, bcause t1 should not be with the 90 degrees anbgle
    t1 = _tav_abs(theta1, refractive)
    theta2 = 40. # this might be a parameter to float
    t2 = _tav_abs(theta2, refractive)
    # 'refractive' is the n parameter in the paper
    x1 = 1 - t1
    x2 = t1**2 * tau**2 * (refractive**2 - t1)
    x3 = t1**2 * tau * refractive**2
    x4 = refractive**4 - tau**2 * (refractive**2 - t1)**2
    x5 = t2 / t1
    x6 = x5 * (t1 - 1) + 1 - t2
    r = x1 + x2 / x4
    t = x3 / x4
    ra = x5 * r + x6
    ta = x5 * t

    # Reflectance and transmittance of N layers
    delta = (t**2 - r**2 - 1)**2 - 4 * r**2
    beta = (1 + r**2 - t**2 - np.sqrt(delta)) / (2 * r)
    va = (1 + r**2 - t**2 + np.sqrt(delta)) / (2 * r)
    vb = (beta * (va - r) / (va * (beta - r)))**0.5
    s1 = ra * (va * vb**(N - 1) - va**(-1) * vb**(-(N - 1))) + (ta * t - ra * r) * (vb**(N - 1) - vb**(-(N - 1)))
    s2 = ta * (va - va**(-1))
    s3 = va * vb**(N - 1) - va**(-1) * vb**(-(N - 1)) - r * (vb**(N - 1) - vb**(-(N - 1)))
    RN = s1 / s3
    TN = s2 / s3

    return RN, TN
