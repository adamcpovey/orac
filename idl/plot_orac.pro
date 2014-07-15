PRO PLOT_ORAC, inst, rev, fdr, stop=stop, compare=comp, preproc=preproc, prev_revision=old, root=root, xsize=xs, ysize=ys, nx=nx, ny=ny, font_size=font_s, scale=scale, label=label, relative=rel, lines=lines, keep_ps=keep_ps, ice=ice, secondary=secondary
   ON_ERROR, 0
   COMPILE_OPT LOGICAL_PREDICATE, STRICTARR, STRICTARRSUBS

   ;; process inputs
   if ~KEYWORD_SET(fdr) then fdr=GETENV('TESTOUT')
   revision=STRING(rev,format='(i0)')
   if SIZE(inst,/type) ne 7 then MESSAGE,'INSTRUMENT must be a string.'
   kc=KEYWORD_SET(comp)
   if ~KEYWORD_SET(xs) then xs=28.3
   if ~KEYWORD_SET(ys) then ys=19.6
   if ~KEYWORD_SET(nx) then nx=4
   if ~KEYWORD_SET(ny) then ny=3
   if ~KEYWORD_SET(font_s) then font_s=8.
   if KEYWORD_SET(scale) then begin
      xs*=scale
      ys*=scale
   endif
   SET_PLOT,'ps'
   
   if KEYWORD_SET(root) then begin
      ;; root filenames provided as arguments
      nroot=N_ELEMENTS(root)
      folder=STRMID(root[0],0,STRPOS(root[0],'/',/reverse_search))
      if kc then begin
         if N_ELEMENTS(comp) ne nroot then $
            MESSAGE,'Mismatched root filename arrays given.'
         oldroot=comp
      endif
   endif else begin
      ;; find root folder for requested revision
      ver=FILE_SEARCH(fdr+'/V*',/test_dir,count=nver)
      if nver le 0 then MESSAGE,'No data found. Please check FOLDER.'
      ver_num=LONG(STRMID(ver,TRANSPOSE(STRPOS(ver,'V',/reverse_search))+1))
      s=SORT(ver_num)
      p=WHERE(ver_num[s] eq rev,np)
      if np ne 1 then MESSAGE,'Requested revision not be found.'
      folder=ver[s[p[0]]]+'/'+inst+'/'

      ;; find root filenames (assuming location preproc file exists)
      root=FILE_SEARCH(folder+'*.loc.nc',count=nroot)
      if nroot le 0 then MESSAGE,'No .LOC.NC files could be found.'
      for i=0,nroot-1 do root[i]=STRMID(root[i],0,STRLEN(root[i])-7)

      if ~KEYWORD_SET(label) then label=STRING(ver_num[s[p[0]]],format='(i0)')

      if kc then begin
         if KEYWORD_SET(old) then begin
            ;; find root folder for requested previous revision
            p=WHERE(ver_num[s] eq old,np)
            if np ne 1 then MESSAGE,'Requested previous revision not be found.'
            oldfolder=ver[s[p[0]]]+'/'+inst+'/'
         endif else begin
            ;; find previous version
            if p[0]-1 lt 0 then MESSAGE,'No previous revision found.'
            oldfolder=ver[s[p[0]-1]]+'/'+inst+'/'
         endelse

         ;; find previous root filenames
         oldroot=FILE_SEARCH(oldfolder+'*.loc.nc',count=noroot)
         if noroot ne nroot then MESSAGE,'Mismatched revisions compared.'
         for i=0,nroot-1 do $
            oldroot[i]=STRMID(oldroot[i],0,STRLEN(oldroot[i])-7)
      endif
   endelse

   if KEYWORD_SET(preproc) then begin
      ;; plot preprocessor outputs
      tag='.preproc'
      suff='.' + ['msi','clf','lsf','alb','geo','loc', $
                  'prtm','lwrtm','swrtm','uv','config'] + '.nc'

      ;; fetch lat/lon for both data grids
      fid=NCDF_OPEN(root[0]+'.loc.nc')
      lat=NCDF_OBTAIN(fid,'lat')
      lon=NCDF_OBTAIN(fid,'lon')
      NCDF_CLOSE,fid

      fid=NCDF_OPEN(root[0]+'.prtm.nc')
      lat_rtm=NCDF_OBTAIN(fid,'lat_pw')
      lon_rtm=NCDF_OBTAIN(fid,'lon_pw')
      i_rtm=NCDF_OBTAIN(fid,'i_pw')
      j_rtm=NCDF_OBTAIN(fid,'j_pw')
      NCDF_CLOSE,fid

      ;; determine field sizes (for chunked plotting)
      sze=SIZE(lat,/dim)
      nl1=sze[0]
      line1=sze[1]
      nl2=MAX(i_rtm)
      line2=MAX(j_rtm)
   endif else begin
      ;; plot ORAC retrieval
      tag='.orac'
      suff = (KEYWORD_SET(ice) ? 'ICE' : 'WAT') + $
             (KEYWORD_SET(secondary) ? (['.primary.nc','.secondary.nc']) : $
             (['.primary.nc']))

      ;; fetch lat/lon
      fid=NCDF_OPEN(root[0]+'WAT.primary.nc')
      lat=NCDF_OBTAIN(fid,'lat')
      lon=NCDF_OBTAIN(fid,'lon')
      qcf=NCDF_OBTAIN(fid,'qcflag')
      NCDF_CLOSE,fid

      ;; determine field size (for chunked plotting)
      sze=SIZE(lat,/dim)
      nl1=sze[0]
      line1=sze[1]
   endelse
   if kc then tag+='.comp'

   ;; save current colourbar and set greyscale for plot titles
   TVLCT, save_ct, /get
   LOADCT,0,/silent

   ;; form plot settings structure
   plot_set = {tag:'', label:'', sheet:-1, font_s:font_s, xs:xs, ys:ys, cs:0., $
               nx:nx, ny:ny, x0:FLTARR(nx), x1:FLTARR(nx), y0:FLTARR(ny), $
               y1:FLTARR(ny), gridi:0, gridj:0, col:0}

   ;; determine character size in PS plot
   DEVICE, /encapsulated, font_s=plot_set.font_s, $
           xsize=plot_set.xs, ysize=plot_set.ys, $
           filen='plot_preproc_test.eps'
   PLOT,[0,1]
   plot_set.cs=plot_set.ys*!d.y_ch_size/!d.y_size
   DEVICE,/close
   FILE_DELETE,'plot_preproc_test.eps'

   ;; determine plot grid in normalised coordinates
   sx=9.0*plot_set.cs           ; horizontal padding around plot
   sy=5.0*plot_set.cs           ; vertical padding around plot
   tx=(plot_set.xs - sx*plot_set.nx) / plot_set.nx ; width of a plot
   ty=(plot_set.ys - sy*plot_set.ny) / plot_set.ny ; height of a plot
   plot_set.x0=(7.0*plot_set.cs + $
                (tx+sx)*FINDGEN(plot_set.nx))/plot_set.xs ; left edge
   plot_set.x1=plot_set.x0 + tx/plot_set.xs               ; right edge
   plot_set.y0=(1.5*plot_set.cs + (ty+sy)* $
                (plot_set.ny-1.-FINDGEN(plot_set.ny)))/plot_set.ys ; bottom edge
   plot_set.y1=plot_set.y0 + ty/plot_set.ys                        ; top edge

   ;; loop over chunks
   for i=0,nroot-1 do begin
      ;; loop over files
      for j=0,N_ELEMENTS(suff)-1 do begin
         if ~FILE_TEST(root[i]+suff[j],/regular) then CONTINUE
         plot_set.tag = root[i]+tag

         ;; open file
         fid=NCDF_OPEN(root[i]+suff[j])
         if kc then fid2=NCDF_OPEN(oldroot[i]+suff[j])
         inq=NCDF_INQUIRE(fid)

         ;; loop over variables
         set=PLOT_SETTINGS(suff[j], inst)
         for k=0,N_ELEMENTS(set)-1 do begin
            ;; read values
            data=NCDF_OBTAIN(fid, set[k].name, fill)
            if kc then data2=NCDF_OBTAIN(fid2, set[k].name, fill2)

            ;; apply requested filter
            case set[k].filter of
               2:    filt = qcf eq 0
               1:    filt = ~(qcf AND 64)
               else: filt = REPLICATE(1,SIZE(data,/dim))
            endcase

            ;; filter out missing values
            if FINITE(fill) then begin
               filt = filt AND data ne fill
               if kc then filt = filt AND data2 ne fill2
            endif else begin
               filt = filt AND FINITE(data)
               if kc then filt = filt AND FINITE(data2)
            endelse
 
            ;; set output if plotting a comparison
            plot_set.col=0
            if kc then begin
               pq=WHERE(filt)
               if ARRAY_EQUAL(data[pq],data2[pq]) $
               then plot_set.col=100 $
               else begin
                  ;; make a difference plot, overriding plot settings
                  set[k].abs=1
                  set[k].log=0
                  set[k].full=0
                  set[k].nlevels=250
                  set[k].btf='(g0.4)'
                  set[k].range[0] = !values.f_nan
                  if KEYWORD_SET(rel) then data=DOUBLE(data)/DOUBLE(data2)-1d0 $
                  else data=DOUBLE(data)-DOUBLE(data2)
               endelse
            endif

            ;; select appropriate plot
            case set[k].mode of
               0: begin
                  ;; if at end of page, start new sheet
                  PLOT_POSITION, plot_set, pos, bpos

                  ;; determine range for plot colourbar
                  ran = FINITE(set[k].range[0]) ? set[k].range : $
                        SELECT_RANGE(data[WHERE(filt)], set[k])

                  ;; print title
                  XYOUTS,/normal,align=0.5,.5*(pos[0]+pos[2]), $
                         color=plot_set.col, $
                         pos[3] + (set[k].mode gt 0 ? 2.:.5)* $
                                  !d.y_ch_size/!d.y_size, $
                         set[k].title eq '' ? FMT(set[k].name) : set[k].title

                  ;; plot line
                  PLOT,data,/noerase,position=pos,color=plot_set.col, $
                       xrange=[0,N_ELEMENTS(data)-1],xstyle=1, $
                       yrange=ran,ystyle=1,ylog=set[k].log,psym=-4
               end
               1: WRAP_MAPPOINTS, data, lat, lon, $
                                  set[k],plot_set,filt,line1,nl1,0.1
               2: for l=0,N_ELEMENTS(data[0,0,*])-1 do $
                  WRAP_MAPPOINTS, data[*,*,l], lat, lon, $
                                  set[k],plot_set,filt[*,*,l],line1,nl1,0.1,l
               3: WRAP_MAPPOINTS, data, lat_rtm, lon_rtm, $
                                  set[k],plot_set,filt,line2,nl2,0.5
               4: for l=0,N_ELEMENTS(data[*,0])-1 do $
                  WRAP_MAPPOINTS, data[l,*], lat_rtm, lon_rtm, $
                                  set[k],plot_set,filt[l,*],line2,nl2,0.5,l
               5: for l=0,N_ELEMENTS(data[*,0])-1,20 do $
                  WRAP_MAPPOINTS, data[l,*], lat_rtm, lon_rtm, $
                                  set[k],plot_set,filt[l,*],line2,nl2,0.5,l
               6: for m=0,N_ELEMENTS(data[*,0,0])-1 do $
                  for l=0,N_ELEMENTS(data[0,*,0])-1,20 do $
                  WRAP_MAPPOINTS, data[m,l,*], lat_rtm, lon_rtm, $
                                  set[k],plot_set,filt[m,l,*],line2,nl2,0.5,l,m
            endcase
         endfor
      endfor 
   endfor
   DEVICE,/close
   SET_PLOT,'x'
   TVLCT, save_ct

   ;; merge individual PDFs and delete pieces
   CD,folder,current=cur_cd
   SPAWN,'gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE='+root[0]+tag+'.pdf '+ $
         '-c "<< /PageSize ['+STRING(ROUND(xs*170./6),format='(i0)')+' '+ $
         STRING(ROUND(ys*170./6),format='(i0)')+'] >> setpagedevice" '+ $
         '-dBATCH '+root[0]+tag+'.N[0-9][0-9].eps > /dev/null'
   if ~KEYWORD_SET(keep_ps) then $
      FILE_DELETE,FILE_SEARCH(root[0]+tag+'.N[0-9][0-9].eps')
   CD,cur_cd

   if KEYWORD_SET(stop) then STOP
END
