macro PlotAnisoCS
{
' ellipsoid
normals_plot = If(modx < 2, sizeL*(3/4),
' cylinder
If(modx < 3,
1/((((Abs((1/cx)*cosphirot))^(10))+((Abs((sinphirot))^(10))*
(((Abs((1/bx)*sinthetarot))^(2)) + ((Abs((1/ax)*costhetarot))^
(2)))^(5)))^(1/10)),
' cuboid
1/((((Abs((1/cx)*cosphirot))^(10))+((Abs((sinphirot))^(10))*
(((Abs((1/bx)*sinthetarot))^(10)) + ((Abs((1/ax)*costhetarot))
^(10)))))^(1/10))
));}

macro AnisoCS( mod, h11, k11, l11, h22, k22, l22, ac, av, bc, bv, cc, cv, pc, pv, tc, tv, nc, nv, sp)
{
	' As published in Ectors et al. (2015) J. Appl. Cryst. 48, 189-194 
	' mod which model to use 1 for ellipsoid 2 for elliptic cylinder 3 for cuboid
	' h11 k11 l11 vector defining the z-axis
	' h22 k22 l22 vector defining the x-axis
	' av bv cv rx ry rz of the model
	' pv tv nv additional rotation parameters theta' theta' nu
	' sp activate special method for triclinic systems 1/0 on/off
' --------- Code starts here ------
		peak_buffer_step 0
	'Initialize variables 
   		#m_argu mod
		#m_argu ac If_Prm_Eqn_Rpt(ac, av, min 0.0001 max 10000)
		#m_argu bc If_Prm_Eqn_Rpt(bc, bv, min 0.0001 max 10000)
		#m_argu cc If_Prm_Eqn_Rpt(cc, cv, min 0.0001 max 10000)
		#m_argu pc If_Prm_Eqn_Rpt(pc, pv, min 0 max 180)
		#m_argu tc If_Prm_Eqn_Rpt(tc, tv, min 0 max 180)
		#m_argu nc If_Prm_Eqn_Rpt(nc, nv, min 0 max 360)
	'G-Metrictensor just for output
		local !m11 = Get(a)^2; local !m12 = Get(a)*Get(b)*Cos(Get(ga)*Deg); local !m13 = Get(a)*Get(c)*Cos(Get(be)*Deg);
		local !m21 = m12;      local !m22 = Get(b)^2;                       local !m23 = Get(b)*Get(c)*Cos(Get(al)*Deg); 
		local !m31 = m13;      local !m32 = m23;                            local !m33 = Get(c)^2;                                    
	'G*-Reciprocal metrictensor and reciprocal lattice parameters
		local !vol = Get(a)*Get(b)*Get(c)*Sqrt(1-((Cos(Get(al)*Deg))^2)-((Cos(Get(be)*Deg))^2)-((Cos(Get(ga)*Deg))^2)+2*(Cos(Get(al)*Deg)*Cos(Get(be)*Deg)*Cos(Get(ga)*Deg)));
		local !ar = (Get(b)*Get(c)*Sin(Get(al)*Deg))/vol;
		local !br = (Get(a)*Get(c)*Sin(Get(be)*Deg))/vol;
		local !cr = (Get(b)*Get(a)*Sin(Get(ga)*Deg))/vol;
		local !alr = (Cos(Get(be)*Deg)*Cos(Get(ga)*Deg)-Cos(Get(al)*Deg))/(Sin(Get(be)*Deg)*Sin(Get(ga)*Deg));
		local !ber = (Cos(Get(al)*Deg)*Cos(Get(ga)*Deg)-Cos(Get(be)*Deg))/(Sin(Get(al)*Deg)*Sin(Get(ga)*Deg));
		local !gar = (Cos(Get(be)*Deg)*Cos(Get(al)*Deg)-Cos(Get(ga)*Deg))/(Sin(Get(be)*Deg)*Sin(Get(al)*Deg));	
		local !r11 = ar^2; local !r12 = ar*br*gar;   local !r13 = ar*cr*ber;
		local !r21 = r12;  local !r22 = br^2;        local !r23 = br*cr*alr;
		local !r31 = r13;  local !r32 = r23;         local !r33 = cr^2; 
	'Test if special method triclinic is activated
 		#m_ifarg sp 1
		local !h1 = 0; local !h2 = If(alr <= 0.00001, 0 , 1);
		local !k1 = 0; local !k2 = If(alr <= 0.00001, 1 , -(r31 / r32));
		local !l1 = 1; local !l2 = 0;
		#m_else
		local !h1 = h11; local !h2 = h22;
		local !k1 = k11; local !k2 = k22;
		local !l1 = l11; local !l2 = l22;
		#m_endif
	'|hkl|Prim aka z-axis
		local rowh1 = (h1 r11 + k1 r12 + l1 r13)*h1; 
		local rowk1 = (h1 r21 + k1 r22 + l1 r23)*k1;
		local rowl1 = (h1 r31 + k1 r32 + l1 r33)*l1; 
		local dvec1 = Sqrt(rowh1+rowk1+rowl1);
	'|hkl|Sec aka x-axis
		local rowh2 = (h2 r11 + k2 r12 + l2 r13)*h2;
		local rowk2 = (h2 r21 + k2 r22 + l2 r23)*k2;
		local rowl2 = (h2 r31 + k2 r32 + l2 r33)*l2;
		local dvec2 = Sqrt(rowh2+rowk2+rowl2); 
	'|hkl|Current
		local rowH = (H r11 + K r12 + L r13)*H; 
		local rowK = (H r21 + K r22 + L r23)*K; 
		local rowL = (H r31 + K r32 + L r33)*L; 
		local dvec3 = Sqrt(rowH+rowK+rowL); 
	'Angle phi
		local rowg1 = (H r11 + K r12 + L r13)*h1; 
		local rowg2 = (H r21 + K r22 + L r23)*k1; 
		local rowg3 = (H r31 + K r32 + L r33)*l1;
		local sum1 = rowg1+rowg2+rowg3; 
		local cosphi = (sum1/(dvec1*dvec3));
		local sinphi = Sqrt(Abs(1-(cosphi^2)));  
	'Angle theta
		local rowp1 = (H r11 + K r12 + L r13)*h2; 
		local rowp2 = (H r21 + K r22 + L r23)*k2; 
		local rowp3 = (H r31 + K r32 + L r33)*l2;
		local sum2 = rowp1+rowp2+rowp3;
		local coseta =(sum2/(dvec2*dvec3));
		local costheta = If(sinphi <= 0.00001,  1, coseta/sinphi);
		local sintheta = Sqrt(Abs(1-(costheta^2)));
	'Handedness
		local p = h1*(k2*L - l2*K) + k1*(l2*H - h2*L) + l1*(h2*K - k2*H);
		local pre = If (p <= 0 , -1 , 1); 
	'Rotation: Rotationmatrix and calculation of phirot and thetarot
		local grot11 = (Cos(CeV(tc,tv) * Deg)*Cos(CeV(pc,pv) * Deg)*Cos(CeV(nc,nv) * Deg))-(Sin(CeV(tc,tv) * Deg)*Sin(CeV(nc,nv) * Deg));
		local grot21 = -(Cos(CeV(tc,tv) * Deg)*Cos(CeV(pc,pv) * Deg)*Sin(CeV(nc,nv) * Deg))-(Sin(CeV(tc,tv) * Deg)*Cos(CeV(nc,nv) * Deg));
		local grot31 = Cos(CeV(tc,tv) * Deg)*Sin(CeV(pc,pv) * Deg);
		local grot12 = (Sin(CeV(tc,tv) * Deg)*Cos(CeV(pc,pv) * Deg)*Cos(CeV(nc,nv) * Deg))+(Cos(CeV(tc,tv) * Deg)*Sin(CeV(nc,nv) * Deg));
		local grot22 = -(Sin(CeV(tc,tv) * Deg)*Cos(CeV(pc,pv) * Deg)*Sin(CeV(nc,nv) * Deg))+(Cos(CeV(tc,tv) * Deg)*Cos(CeV(nc,nv) * Deg));
		local grot32 = Sin(CeV(tc,tv) * Deg)*Sin(CeV(pc,pv) * Deg);
		local grot13 = -(Sin(CeV(pc,pv) * Deg)*Cos(CeV(nc,nv) * Deg));
		local grot23 = Sin(CeV(pc,pv) * Deg)*Sin(CeV(nc,nv) * Deg);
		local grot33 = Cos(CeV(pc,pv) * Deg);
		local cosphirot = (costheta*sinphi*grot31)+(pre*sintheta*sinphi*grot32)+(cosphi*grot33);
		local sinphirot = Sqrt(Abs(1-(cosphirot^2)));
		local xfac = (costheta*sinphi*grot11)+(pre*sintheta*sinphi*grot12)+(cosphi*grot13);
		local costhetarot = If(sinphirot <= 0.0001, 1, xfac / sinphirot);
		local sinthetarot = Sqrt(Abs(1-(costhetarot^2)));
	'Models:
	'Ellipsoid
		#m_ifarg mod 1
		local volop  = ((4/3) 3.14159265358979 CeV(ac,av) CeV(bc,bv) CeV(cc,cv));
		local areaop = 3.14159265358979 Sqrt(((CeV(cc,cv) sinphirot)^2)(((CeV(ac,av) sinthetarot)^2)+(CeV(bc,bv) costhetarot)^2)+((CeV(ac,av) CeV(bc,bv) cosphirot)^2));
		local sizeL = volop / areaop;
		#m_endif
	'Cylinder
		#m_ifarg mod 2
		local volop  = (3.14159265358979 CeV(ac,av) CeV(bc,bv) (2 * CeV(cc,cv)));
		local areaop = (Sqrt( ((CeV(ac,av) sinthetarot)^2) + ((CeV(bc,bv) costhetarot)^2)) 4 * CeV(cc,cv) * Abs(sinphirot)) + (3.14159265358979 CeV(ac,av) CeV(bc,bv) Abs(cosphirot));
		local sizeL = volop / areaop;
		#m_endif
	'Cuboid
		#m_ifarg mod 3
		local volop  = (8 CeV(ac,av) CeV(bc,bv) CeV(cc,cv));
		local areaop = 4 CeV(ac,av) CeV(bc,bv) Abs(cosphirot) + 4 CeV(ac,av) CeV(cc,cv) Abs(sinphirot) Abs(sinthetarot) + 4 CeV(bc,bv) CeV(cc,cv) Abs(costhetarot) Abs(sinphirot) ;
		local sizeL = volop / areaop;
		#m_endif
	'For Output purposes only
		local ax = CeV(ac,av);
		local bx = CeV(bc,bv);
		local cx = CeV(cc,cv);
		local px = CeV(pc,pv);
		local tx = CeV(tc,tv);
		local nx = CeV(nc,nv);
		local modx = mod; 
		local vole = volop;
		local larea = volop^(1/3);
		local kk = larea / sizeL ;
		local areaoe = areaop;
		local phi = If(cosphi < -1, 180, If (cosphi > 1, 0, ArcCos(cosphi)*Rad));
		local the = If(costheta < -1, 180, If (costheta > 1, 0, ArcCos(costheta)*Rad));
		local phirot = If(cosphirot < -1, 180, If (cosphirot > 1, 0, ArcCos(cosphirot)*Rad));
		local therot = If(costhetarot < -1, 180, If (costhetarot > 1, 0, ArcCos(costhetarot)*Rad));
	'CS_Macro
	        'lor_fwhm = 0.1 180/pi^2 * Lambda / Cos(Th) * sizeL ... 0.1 for nm->Angstroem
		lor_fwhm = 0.1 18.2378130556208 Lam / (Cos(Th) sizeL); 
}

macro AnisoCSout(file) {
' As published in Ectors et al. (2015) J. Appl. Cryst. 48, 189-194 
out file
Out_String("Analysis Report: ")
Out(Get(phase_name),"%s \n")
Out(Get (r_wp), " Rwp: %8.3f \n")
Out(Get (r_exp), " Rexp:%8.3f \n\n")

Out_String("G-Metrictensor \n\n")
Out(m11,"%8.3f")	Out(m12,"%8.3f")	Out(m13,"%8.3f\n")
Out(m21,"%8.3f")	Out(m22,"%8.3f")	Out(m23,"%8.3f\n")
Out(m31,"%8.3f")	Out(m32,"%8.3f")	Out(m33,"%8.3f\n\n")

Out_String("G*-Metrictensor \n\n")
Out(r11,"%8.3f")	Out(r12,"%8.3f")	Out(r13,"%8.3f\n")
Out(r21,"%8.3f")	Out(r22,"%8.3f")	Out(r23,"%8.3f\n")
Out(r31,"%8.3f")	Out(r32,"%8.3f")	Out(r33,"%8.3f\n\n")

Out_String("Rotationmatrix \n\n")
Out(grot11,"%8.3f")	Out(grot12,"%8.3f")	Out(grot13,"%8.3f\n")
Out(grot21,"%8.3f")	Out(grot22,"%8.3f")	Out(grot23,"%8.3f\n")
Out(grot31,"%8.3f")	Out(grot32,"%8.3f")	Out(grot33,"%8.3f\n\n")


Out_String("Model parameters \n")
Out_String(" (1) Ellipsoid / (2) Cylinder / (3) Cuboid \n")
Out(modx, " Model: %1.0f \n\n")
Out(h1," z-axis hkl: %8.3f")	Out(k1," %8.3f")	Out(l1," %8.3f \n")
Out(px, " Rotation phi: %8.3f") Out(tx, "  theta: %8.3f \n\n")
Out(h2," x-axis hkl: %8.3f")	Out(k2," %8.3f")	Out(l2," %8.3f \n")
Out(nx, " Additional rotation nu: %8.3f \n\n")
Out(ax," rx-radius: %8.3f nm \n")
Out(bx," ry-radius: %8.3f nm \n")
Out(cx," rz-radius: %8.3f nm \n")
Out(vole," Volume: %8.3f nm^3 \n")
Out(larea," <True CS>: %8.3f nm \n\n")



      Out_String("   H   K  L   M   2Th        D        phi     theta    phirot  thetarot  <L_surf>\n")
phase_out file append load out_record out_fmt out_eqn 

{
            "%4.0f" = H;
            "%4.0f" = K;
            "%4.0f" = L;
            "%4.0f" = M;
            "%8.3f" = 2 Th Rad;
            "%8.3f" =  D_spacing;
            "%8.3f" = phi;
            "%8.3f" = the;
            "%8.3f" = phirot;
            "%8.3f" = therot;
            "%8.3f\n" = sizeL;
   }

}

macro AnisoCSg( tauc, tauv)
{
		#m_argu tauc If_Prm_Eqn_Rpt(tauc, tauv, min 1 max 2)
	'Parameters Eq.(5-7)
		local para = Abs((costhetarot*sinphirot)/ax);
		local parb = Abs((sinthetarot*sinphirot)/bx);
		local parc = Abs((cosphirot)/cx);
	'Parameters chapter 2.4
		local pard = Sqrt(para^2+parb^2);
		local pare = pard/parc;
		local parf = Sqrt(1-pare^2);
	'Here comes the tricky part
	'Ellipsoid Eq.(4)
		local ev = If(modx < 2,CeV(tauc,tauv) * (9/8) * sizeL,
	'Cylinder 
		If(modx < 3,
	'Eq. (11)
		If(pard <= 0.0001,CeV(tauc, tauv) * sizeL,
	'Eq. (12)
		If(pard < parc, CeV(tauc, tauv) * ( (2/parc) - (8/(3.14159265358979*pard)) * (-(2/3)+(2/3)*parf+((pare*pare*parf)/3)+(pare*ArcSin(pare)))+
		((2*parc)/(3.14159265358979*pard*pard))*(((pare*parf)/2)-((1/2)*(ArcSin(pare)))+(pare*pare*pare*parf)+(2*pare*pare*ArcSin(pare)))),
	'Eq. (13)
		 CeV(tauc, tauv)*((16)/(3*3.14159265358979*pard))-((2*parc)/(4*pard*pard)))),
	'Cuboid
	'Eq. (8)
		If(And(para>=parb, para>=parc),CeV(tauc, tauv)*(2/para)*(1-(parb/(3*para))-(parc/(3*para))+((parb*parc)/(6*para*para))),
	'Eq. (9)
		If(And(parb>=para, parb>=parc),CeV(tauc, tauv)*(2/parb)*(1-(para/(3*parb))-(parc/(3*parb))+((para*parc)/(6*parb*parb))),
	'Eq. (10)
		CeV(tauc, tauv)*(2/parc)*(1-(para/(3*parc))-(parb/(3*parc))+((para*parb)/(6*parc*parc))))))
		);
	'IB calculations Eq. (1-3)
		local ibV = 1/ev;
		local ibL = 1/(2*sizeL);
	'ibG is 0 if ibV < ibL 
		local ibG = If (ibV > ibL, ((Sqrt((4*ibV^2)-(4.27679864*ibV*ibL)+(0.27679864*ibL^2)))/2), 0);
	'CS_G macro
		local sizeG = ibG/(Sqrt(3.14159265358979/(4*Ln(2))));
		gauss_fwhm = 0.1 Rad Lam sizeG / Cos(Th);
	'For Output purposes only
		local taux =  CeV(tauc, tauv);
}

macro AnisoCSgout(file) {
out file
Out_String("Analysis Report: ")
Out(Get(phase_name),"%s \n")
Out(Get (r_wp), " Rwp: %8.3f \n")
Out(Get (r_exp), " Rexp:%8.3f \n\n")

Out_String("G-Metrictensor \n\n")
Out(m11,"%8.3f")	Out(m12,"%8.3f")	Out(m13,"%8.3f\n")
Out(m21,"%8.3f")	Out(m22,"%8.3f")	Out(m23,"%8.3f\n")
Out(m31,"%8.3f")	Out(m32,"%8.3f")	Out(m33,"%8.3f\n\n")

Out_String("G*-Metrictensor \n\n")
Out(r11,"%8.3f")	Out(r12,"%8.3f")	Out(r13,"%8.3f\n")
Out(r21,"%8.3f")	Out(r22,"%8.3f")	Out(r23,"%8.3f\n")
Out(r31,"%8.3f")	Out(r32,"%8.3f")	Out(r33,"%8.3f\n\n")

Out_String("Rotationmatrix \n\n")
Out(grot11,"%8.3f")	Out(grot12,"%8.3f")	Out(grot13,"%8.3f\n")
Out(grot21,"%8.3f")	Out(grot22,"%8.3f")	Out(grot23,"%8.3f\n")
Out(grot31,"%8.3f")	Out(grot32,"%8.3f")	Out(grot33,"%8.3f\n\n")


Out_String("Model parameters \n")
Out_String(" (1) Ellipsoid / (2) Cylinder / (3) Cuboid \n")
Out(modx, " Model: %1.0f \n\n")
Out(h1," z-axis hkl: %8.3f")	Out(k1," %8.3f")	Out(l1," %8.3f \n")
Out(px, " Rotation phi: %8.3f") Out(tx, "  theta: %8.3f \n\n")
Out(h2," x-axis hkl: %8.3f")	Out(k2," %8.3f")	Out(l2," %8.3f \n")
Out(nx, " Additional rotation nu: %8.3f \n\n")
Out(ax," rx-radius_surf: %8.3f nm ")
Out(ax*taux,"/ rx-radius_vol: %8.3f nm \n")
Out(bx," ry-radius_surf: %8.3f nm ")
Out(bx*taux,"/ ry-radius_vol: %8.3f nm \n")
Out(cx," rz-radius_surf: %8.3f nm ")
Out(cx*taux,"/ rz-radius_vol: %8.3f nm \n")
Out(vole," Volume_surf: %8.3f nm^3 ")
Out(vole*taux*taux*taux,"/ Volume_vol: %8.3f nm^3 \n")
Out(larea," <True CS>_surf: %8.3f nm ")
Out(larea*taux,"/ <True CS>_vol: %8.3f nm \n\n")
Out_String("Distribution parameters \n")
Out(taux," Tau: %8.4f  \n")
Out_String(" If Lognormal of <True CS>\n")
Out(Ln(larea*taux)-(3.5*Ln(taux))," Logn. mean:%8.4f  \n")
Out(Sqrt(Ln(taux))," Logn. var.: %8.4f  \n\n")

      Out_String("   H   K  L   M   2Th        D    <L_surf><L_vol>  <L_vol>/<L_surf> \n")
phase_out file append load out_record out_fmt out_eqn 

{
            "%4.0f" = H;
            "%4.0f" = K;
            "%4.0f" = L;
            "%4.0f" = M;
            "%8.3f" = 2 Th Rad;
            "%8.3f" =  D_spacing;
            "%8.3f" = sizeL;
            "%8.3f" = ev;
            "%8.3f\n" = ev/sizeL;
   }

}