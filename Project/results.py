#Simple model

simple_overall_mse =  [115862190000.0]
simple_test_mses =  [230325108736.0, 128924016640.0, 118739664896.0, 116505870336.0, 116333936640.0, 116382269440.0, 116183752704.0, 116049240064.0, 116097851392.0, 116103405568.0, 115916029952.0, 116025344000.0, 115899736064.0, 115937083392.0, 115911229440.0, 115870490624.0, 115868598272.0, 116024254464.0, 115897483264.0, 115956047872.0, 116022534144.0, 115968262144.0, 115876102144.0, 115871432704.0, 115862192128.0]


simple_train_mses = [776884410210.4615, 162645271158.15384, 130546044928.0, 124534074761.84616, 123570565750.15384, 123591094272.0, 123426757080.61539, 123377645410.46153, 123484898540.3077, 123532241683.6923, 123370930176.0, 123912949444.92308, 123526702631.38461, 123520480177.23077, 123344526099.6923, 123416058328.61539, 123335246139.07692, 123561009467.07692, 123406662734.76923, 123588337979.07692, 123344278764.3077, 123552330988.3077, 123491970756.92308, 123681244396.3077, 123581128388.92308]


simple_val_mses = [771183916898.4615, 149721315170.46155, 122440670916.92308, 117097618510.76923, 116407990114.46153, 116313142823.38461, 116261895246.76923, 116234197464.61539, 116229695645.53847, 116200518892.3077, 116216348672.0, 116200575291.07692, 116206886596.92308, 116217860726.15384, 116191027515.07692, 116201930121.84616, 116222044632.61539, 116229745112.61539, 116187095985.23077, 116218989331.6923, 116185980612.92308, 116219360492.3077, 116236562432.0, 116254138683.07692, 116364894208.0]



#Complete final model with dropout and relu

complete_overall_mse = [81730540000.0]
complete_test_mses = [231861370880.0, 128447266816.0, 106236772352.0, 86392586240.0, 85244461056.0, 86571106304.0, 86269927424.0, 84877680640.0, 84657700864.0, 83615629312.0, 82842796032.0, 82794651648.0, 83116695552.0, 82771394560.0, 82434097152.0, 83097214976.0, 82394652672.0, 83903283200.0, 83002523648.0, 84211744768.0, 82530877440.0, 82103345152.0, 82310201344.0, 82214838272.0, 81730543616.0]


complete_train_mses = [236485640192.0, 223124045193.84616, 135678723623.38461, 102621380923.07692, 92412498235.07692, 91292071463.38461, 91359825920.0, 90908611505.23077, 90784591556.92308, 90607946830.76923, 90684908150.15384, 90524653725.53847, 90435382035.6923, 90176771938.46153, 89960050372.92308, 89714014995.6923, 89740253656.61539, 90099394717.53847, 90034300455.38461, 89997395180.3077, 90273527020.3077, 89705697280.0, 89326461558.15384, 89527794451.6923, 89084369683.6923]


complete_val_mses = [222363044942.76923, 207247945097.84616, 126175099352.61539, 95394758656.0, 86399820091.07692, 85551037518.76923, 85409844932.92308, 85024390852.92308, 84922566813.53847, 84844615207.38461, 84886630400.0, 84705489211.07692, 84624338313.84616, 84461558547.6923, 84404802481.23077, 84138030946.46153, 84073752733.53847, 84051319886.76923, 84385754190.76923, 84336778318.76923, 84449614926.76923, 84174411618.46153, 83771005085.53847, 83827648354.46153, 83543984915.6923]



#Final model without Dropout but with relu

complete_nodropout_overall_mse = [82805810000.0]
complete_nodropout_test_mses =  [206457241600.0, 122287448064.0, 98226561024.0, 88635088896.0, 88051105792.0, 84417437696.0, 84147625984.0, 84148215808.0, 85773541376.0, 84405190656.0, 84104232960.0, 83257843712.0, 83612647424.0, 83191029760.0, 83236462592.0, 83889905664.0, 83282698240.0, 83007111168.0, 83597000704.0, 83615825920.0, 83545120768.0, 84293165056.0, 83782631424.0, 83664240640.0, 82805809152.0]


complete_nodropout_train_mses =  [232907286370.46155, 167938967709.53845, 114779434062.76923, 100981867441.23077, 95189204676.92308, 94863455783.38461, 91808716327.38461, 91917813287.38461, 91789134611.6923, 91435143483.07692, 91918656433.23077, 91624636731.07692, 91062399606.15384, 90770850422.15384, 91112847832.61539, 91302585895.38461, 90722212627.6923, 90826028583.38461, 90545118601.84616, 90507347495.38461, 90990690619.07692, 90914258313.84616, 90671970461.53847, 90444266259.6923, 90535270242.46153]


complete_nodropout_val = [218809826067.69232, 154787591719.3846, 107498072851.6923, 94494962924.3077, 89316771997.53847, 88654052115.6923, 86003011268.92308, 85706699697.23077, 85620808782.76923, 85589953299.6923, 85756379136.0, 85838207763.6923, 85217025102.76923, 85210161782.15384, 85348526867.6923, 85560470921.84616, 85069349809.23077, 85062265934.76923, 84941900878.76923, 84994393954.46153, 85377070001.23077, 85218450510.76923, 85116393944.61539, 84781397228.3077, 84978775591.38461]


#Final model with dropout, without relu

complete_nodropout_norelu_overall_mse = [87410065000.0]
complete_nodropout_norelu_test_mses = [151176249344.0, 114526797824.0, 104297119744.0, 98990669824.0, 92053913600.0, 90924097536.0, 92731695104.0, 90284367872.0, 89920077824.0, 89454968832.0, 89558556672.0, 89168297984.0, 90535272448.0, 88219516928.0, 89051004928.0, 91174256640.0, 87644782592.0, 88129691648.0, 88148205568.0, 89450930176.0, 87695826944.0, 88027766784.0, 86975700992.0, 90289029120.0, 87410065408.0]


complete_nodropout_norelu_train_mses =  [200920480531.69232, 134384317361.23077, 112679935684.92308, 104317412588.3077, 100496076484.92308, 98239003096.61539, 98684963288.61539, 98655087694.76923, 96616745747.6923, 96307498062.76923, 95386756332.3077, 94739455054.76923, 94051004416.0, 93679192536.61539, 93752641378.46153, 93349181282.46153, 94028747697.23077, 92845719236.92308, 92624750119.38461, 92531292947.6923, 94616967640.61539, 93182002412.3077, 92349021577.84616, 92540465467.07692, 92293849403.07692]


complete_nodropout_val_test_mses =  [186335859003.07693, 125240478483.6923, 105634015389.53847, 98461668588.3077, 94635760403.6923, 92304069553.23077, 92992016384.0, 92089084691.6923, 90492049565.53847, 90034556612.92308, 88886930195.6923, 88109850308.92308, 87664461351.38461, 87183750695.38461, 87479031020.3077, 87186013262.76923, 87912588209.23077, 86757047059.6923, 86548602564.92308, 86686521186.46153, 88615890313.84616, 87366383143.38461, 86571298185.84616, 86495093681.23077, 86818993388.3077]
