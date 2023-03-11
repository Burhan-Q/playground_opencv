# pycocotools MatLAB translation attempt

def analyze(ev):
    # Derek Hoiem style analyis of false positives.
    outDir='./analyze'
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    if not hasattr(ev.cocoGt.data.annotations, 'ignore'):
        ev.cocoGt.data.annotations.ignore = 0
    dt=ev.cocoDt
    gt=ev.cocoGt
    prm=ev.params
    rs=prm.recThrs
    ev.params.maxDets=100
    catIds=ev.cocoGt.getCatIds()
    # compute precision at different IoU values
    ev.params.catIds=catIds
    ev.params.iouThrs=[.75, .5, .1]
    ev.evaluate()
    ev.accumulate()
    ps=ev.eval.precision
    ps[4:7,:,:,:]=0
    ev.params.iouThrs=.1
    ev.params.useCats=0
    for k in range(len(catIds)):
        catId=catIds[k]
        nm=ev.cocoGt.loadCats(catId)
        nm=[nm.supercategory + '-' + nm.name]
        print('\nAnalyzing %s (%i):\n' %(nm,k))
        clk=time.clock()
        # select detections for single category only
        D=dt.data
        A=D.annotations
        A=A([A.category_id]==catId)
        D.annotations=A
        ev.cocoDt=dt
        ev.cocoDt=CocoApi(D)
        # compute precision but ignore superclass confusion
        is=gt.getCatIds('supNms',gt.loadCats(catId).supercategory)
        D=gt.data
        A=D.annotations
        A=A(ismember([A.category_id],is))
        [A([A.category_id]!=catId).ignore]=deal(1)
        D.annotations=A
        ev.cocoGt=CocoApi(D)
        ev.evaluate()
        ev.accumulate()
        ps[4,:,k,:]=ev.eval.precision
        # compute precision but ignore any class confusion
        D=gt.data
        A=D.annotations
        [A([A.category_id]!=catId).ignore]=deal(1)
        D.annotations=A
        ev.cocoGt=gt
        ev.cocoGt.data=D
        ev.evaluate()
        ev.accumulate()
        ps[5,:,k,:]=ev.eval.precision
        # fill in background and false negative errors and plot
        ps[ps==-1]=0
        ps[6,:,k,:]=ps[5,:,k,:]>0
        ps[7,:,k,:]=1
        makeplot(rs,ps[:,:,k,:],outDir,nm)
        print('DONE (t=%0.2fs).\n' %(time.clock()-clk))
    # plot averages over all categories and supercategories
    ev.cocoDt=dt
    ev.cocoGt=gt
    ev.params=prm
    print('\n')
    makeplot(rs,mean(ps,3),outDir,'overall-all')
    sup={ev.cocoGt.loadCats(catIds).supercategory}
    for k in unique(sup):
        ps1=mean(ps(:,:,strcmp(sup,k),:),3)
        makeplot(rs,ps1,outDir,['overall-' + k[1]])

def makeplot(rs, ps, outDir, nm):
    # Plot FP breakdown using area plot.
    print('Plotting results...                  ')
    t=time.clock()
    cs=[[1, 1, 1], [.31, .51, .74], [.75, .31, .30], [.36, .90, .38], [.50, .39, .64], [1, .6, 0]]
    m=size(ps,1)
    areaNms=['all','small','medium','large']
    nm0=nm
    ps0=ps
    for a in range(size(ps,4)):
        nm=nm0 + '-' + areaNms[a]
        ps=ps0[:,:,:,a]
        ap=round(mean(ps,2)*1000)
        ds=[ps[1,:], diff(ps)]
        ls=['C75','C50','Loc','Sim','Oth','BG','FN']
        for i in range(m):
            if ap(i)==1000:
                ls[i]=['[1.00] ' + ls[i]]
            else:
                ls[i]=sprintf('[.%03i] %s',ap(i),ls[i])
        figure(1)
        clf()
        h=area(rs,ds)
        legend(ls,'location','sw')
        for i in range(m):
            set(h(i),'FaceColor',cs[i])
        title(nm)
        xlabel('recall')
        ylabel('precision')
        set(gca,'fontsize',20)
        nm=[outDir + '/' + regexprep(nm,' ','_')]
        print(nm,'-dpdf')
        [status,~]=system(['pdfcrop ' + nm + '.pdf ' + nm + '.pdf'])
        if status>0:
            warning('pdfcrop not found.')
    print('DONE (t=%0.2fs).\n' %(time.clock()-t))