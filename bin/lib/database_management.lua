
require "./utils.lua";

DB_NAME = "davide"
DB_USER = "dchicco"
DB_PWD = "davide"
DB_ADDRESS = "192.168.2.12"
DATA_NORMALIZATION = false
MAX_MEAN = 200
PRINT_MANIPULATION = true
PRINT_AVERAGES = false

IMR90_COL_INDEX = 57
HUVEC_COL_INDEX = 55
K562_COL_INDEX = 61
GM12878_COL_INDEX = 26


-- Function that reads a cell type and returns the index of the neuron in the neural network
function retrieveNeuronIndexFromCellType(cell_type)
  
  local output_var = ""
  
  if cell_type=="IMR90" then output_var=IMR90_COL_INDEX end
  if cell_type=="HUVEC" then output_var=HUVEC_COL_INDEX end
  if cell_type=="k562" then output_var=K562_COL_INDEX end
  if cell_type=="GM12878" then output_var=GM12878_COL_INDEX end
  
  return output_var
  
end

printOnce = true

-- DATABASE CONNECTION global connection tools
driver = require "luasql.postgres"
env = nil
con = nil

-- Function that checks if a connection to the database is open or not
function isGlobalDbConnectionOpen()
  
  local statusString = ""
  if con==nil then 
    statusString = "closed"
  else
    statusString = (tostring(con):match"closed" and "closed" or "open")
  end
  
  if statusString=="open" then return true;
  elseif statusString=="closed" then return false;
  end
end

-- Function which opens the global connection to the database on the global con and env variables
function openGlobalDbConnection()
  
  if (isGlobalDbConnectionOpen()==true) then
    
    print("The global database connection is already open, so the system will not open it again and will return the current global connection")
    
  else
      
    print("The global database connection is CLOSED, so the system will open it and return it")
    -- create environment object
    env = assert(driver.postgres());
    -- connect to data source  
    con = assert(env:connect(DB_NAME, DB_USER, DB_PWD, DB_ADDRESS));
  end
  
  return con;
  
end  
  

-- Function which closes the global connection to the database on the global con and env variables
function closeGlobalDbConnection() 
  
 if (isGlobalDbConnectionOpen()==true) then
    con:close();
    env:close();
  end
  print("The global database connection is CLOSED")
end


-- Function that takes a chromRegion profile pair, the original dnase table, the indices of the positions, and returns the id's of the selected pair
function fromProfileToChromRegionCoupleId_Thurman2012data(chromSel, profileFirst, profileSecond, dnaseDataTable, first_profile_initial, first_profile_finish, second_profile_initial, second_profile_finish, last_index)
  
  if chromSel==nil then print("chromSel==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if profileFirst==nil then print("profileFirst==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if profileSecond==nil then print("profileSecond==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if dnaseDataTable==nil then print("dnaseDataTable==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if first_profile_initial==nil then print("first_profile_initial==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if first_profile_finish==nil then print("first_profile_finish==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if second_profile_initial==nil then print("second_profile_initial==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if second_profile_finish==nil then print("second_profile_finish==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  if last_index==nil then print("last_index==nil fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
  

  -- if (chromSel==nil or profileFirst==nil or profileSecond==nil or dnaseDataTable==nil or first_profile_initial==nil or first_profile_finish==nil or second_profile_initial==nil or second_profile_finish==nil or last_index or second_profile_finish==nil) then print("fromProfileToChromRegionCoupleId_Thurman2012data() Error, the program will stop"); os.exit(); end
    
  
  local chromName_index = 1
  local first_chromRegionIdIndex = 2
  local second_chromRegionIdIndex = 3
  
   if #dnaseDataTable==0 then print("#dnaseDataTable==0 Error, program will exit."); os.exit() end
   
   

--     io.write("profileFirst: ")
--     for p=1,(#profileFirst)[1] do
--       io.write(profileFirst[p].." ")
--     end
--     io.write("\n");   
--     io.write("profileSecond: ")
--     for p=1,(#profileSecond)[1] do
--       io.write(profileSecond[p].." ")
--     end
--     io.write("\n");
--     --sys.sleep(0.3)
--     
--     print("##################");
    
   
   
  for i=1,#dnaseDataTable,1 do -- number of rows
    rate = round(i*100/#dnaseDataTable,2);
    -- if(i%10==0) then io.write(rate.."% "); io.flush(); end
   
    local tempTens = torch.Tensor(dnaseDataTable[i])
    
    local dataset_firstChromRegionNEW = {}
    local dataset_secondChromRegionNEW = {}
    dataset_firstChromRegionNEW = tempTens[{{first_profile_initial,first_profile_finish}}]
    dataset_secondChromRegionNEW = tempTens[{{second_profile_initial,second_profile_finish}}]

    
--      io.write("i= "..i..") dataset_firstChromRegionNEW: ")
--      for p=1,(#dataset_firstChromRegionNEW)[1] do
--        io.write(dataset_firstChromRegionNEW[p].." ")
--      end
--    
--     io.write("\n");
--     io.write("i= "..i..") dataset_secondChromRegionNEW: ")
--     for p=1,(#dataset_secondChromRegionNEW)[1]  do
--       io.write(dataset_secondChromRegionNEW[p].." ")
--     end
--     io.write("\n");
--     --sys.sleep(0.3)
    
    
--        io.write("(#profileFirst)[1] "..(#profileFirst)[1].."\n")
--        io.write("(#profileSecond[1] "..(#profileSecond)[1].."\n")
--        io.write("(#dataset_firstChromRegionNEW)[1] "..tonumber((#dataset_firstChromRegionNEW)[1]).."\n")
--        io.write("(#dataset_secondChromRegionNEW)[1] "..tonumber((#dataset_secondChromRegionNEW)[1]).."\n")
-- --        os.exit();
   

    if ( (haveTwoTensorsTheSameContent(dataset_firstChromRegionNEW, profileFirst) and
       haveTwoTensorsTheSameContent(dataset_secondChromRegionNEW, profileSecond)) or
       (haveTwoTensorsTheSameContent(dataset_firstChromRegionNEW, profileSecond) and
       haveTwoTensorsTheSameContent(dataset_secondChromRegionNEW, profileFirst)) ) then
      
      -- print("Found INTERACTION");
      
       --io.write("chr"..tempTens[chromName_index].."-"..tempTens[first_chromRegionIdIndex].." ");
       --io.write("and chr"..tempTens[chromName_index].."-"..tempTens[second_chromRegionIdIndex].."\n");
      
     do return {tempTens[first_chromRegionIdIndex], tempTens[second_chromRegionIdIndex]}; end

    end

    collectgarbage();
  end
  
  
  print("\nNo interaction found! Something's wrong!"); os.exit();
  return NULL;
end



--## FUNCTION function that takes a chromRegion profile and returns its chromregion
function fromProfileToChromRegion_Miriam2014(chromSel, profile)

  if GLOBAL_DATA_SOURCE=="Miriam2014" then
    dnaseDataFile = "../data/2015-01-15_Miriam_data/dnaseInput/dnase_"..chromSel.."_without_bashes";

  --elseif CREATE THE FILE FOR Thurman2012 DATA

  end


  local dnaseDataFileReader = io.open(dnaseDataFile, "r");
  local dnaseDataTable = {};
  local chromRegion = {};
  i=1
    for line in dnaseDataFileReader:lines() do

	dnaseDataTable[i] = {}
	dnaseDataTable[i] =  {unpack(line:split(","))}

--     print("\nnew comparison");
--     io.write(profile[1].." vs "); io.write((dnaseDataTable[i][4]).."\n");
--     io.write(profile[2].." vs "); io.write((dnaseDataTable[i][5]).."\n");
--     io.write(profile[3].." vs "); io.write((dnaseDataTable[i][6]).."\n");
--     io.write(profile[4].." vs "); io.write((dnaseDataTable[i][7]).."\n");
--     io.write(profile[5].." vs "); io.write((dnaseDataTable[i][8]).."\n");
--     io.write(profile[6].." vs "); io.write((dnaseDataTable[i][9]).."\n\n");
--     sleep(5);



      idp = 10;

	if dnaseDataTable[i][4]~="gm12878" and areSame(dnaseDataTable[i][4],profile[1])
and areSame(dnaseDataTable[i][5],profile[2])
and areSame(dnaseDataTable[i][6],profile[3])
and areSame(dnaseDataTable[i][7],profile[4])
and areSame(dnaseDataTable[i][8],profile[5])
and areSame(dnaseDataTable[i][9],profile[6])--[[

        and round(tonumber(dnaseDataTable[i][5]),idp) == round(tonumber(profile[2]),idp)
        and round(tonumber(dnaseDataTable[i][6]),idp) == round(tonumber(profile[3]),idp)
        and round(tonumber(dnaseDataTable[i][7]),idp) == round(tonumber(profile[4]),idp)
        and round(tonumber(dnaseDataTable[i][8]),idp) == round(tonumber(profile[5]),idp)
        and round(tonumber(dnaseDataTable[i][9]),idp) == round(tonumber(profile[6]),idp) ]]
	then

	  chromRegion[1] = dnaseDataTable[i][1];
	  chromRegion[2] = dnaseDataTable[i][2];
	  chromRegion[3] = dnaseDataTable[i][3];       
	  
	  -- io.write("Found ");
	  io.write(chromRegion[1] .."-"..chromRegion[2] .."-"..chromRegion[3]);
	  io.flush();
	  break;

	end	

	i = i + 1;
	collectgarbage();
    end
  dnaseDataFileReader.close();


  return chromRegion;

end



-- Function that reads the predictions and the datasets and returns all the details of the predicted interactions
function retrieveInteractionsDetails_Thruman2012data(arrayFPindices, dataset_firstChromRegion, dataset_secondChromRegion, chromSel, dnaseDataTable, first_profile_initial, first_profile_finish, second_profile_initial, second_profile_finish, last_index)
  
  if (arrayFPindices==nil or  dataset_firstChromRegion==nil or  dataset_secondChromRegion==nil or  chromSel==nil or  dnaseDataTable==nil or  first_profile_initial==nil or  first_profile_finish==nil or  second_profile_initial==nil or second_profile_finish==nil or  last_index==nil) then print("retrieveInteractionsDetails_Thruman2012data() Error, program will stop "); os.exit(); end
  
  io.write("retrieveInteractionsDetails_Thruman2012data()\nrate= ");
  io.flush();
  
  local final_interactions = {}

    for i=1,#arrayFPindices do
	 if i%10==0 then rate=round(i*100/#arrayFPindices,2) io.write(rate.."% ") end

	final_interactions[i] = fromProfileToChromRegionCoupleId_Thurman2012data(chromSel, dataset_firstChromRegion[arrayFPindices[i]], dataset_secondChromRegion[arrayFPindices[i]], dnaseDataTable, first_profile_initial, first_profile_finish, second_profile_initial, second_profile_finish, last_index)
	
	io.flush();
	collectgarbage();
    end

    if (final_interactions==nil) then print("retrieveInteractionsDetails_Thruman2012data() for cycle Error, the program will stop"); os.exit();  end
    
    return final_interactions;
end



-- Function that generates the command to retrieve each singular row in then
-- PostgreSQL query
function sqlRowRetrieverCommand(row, dataSource, labelValue, first_profile_initial, second_profile_finish, dnaseExcludeColumnNumber, dnaseExcludeColumnName, numberOfCellTypes)

  local readTensor = {}
  
  if dataSource=="Thurman2012" or dataSource=="Thurman_Miriam" then
    
    readTensor = torch.Tensor({tonumber(row.name), tonumber(row.crp1_id_region), tonumber(row.crp2_id_region), tonumber(row.crp1_a549_ds14289), tonumber(row.crp1_ag10803_ds12374), tonumber(row.crp1_aoaf_ds13513), tonumber(row.crp1_cd14_ds17215), tonumber(row.crp1_cd19_ds17186), tonumber(row.crp1_cd20_ds18208), tonumber(row.crp1_cd34_ds12274), tonumber(row.crp1_cd3_cordblood_ds17706), tonumber(row.crp1_cd3_ds17198), tonumber(row.crp1_cd4_ds17212), tonumber(row.crp1_cd4pos_n_ds14108), tonumber(row.crp1_cd56_ds17189),
	    tonumber(row.crp1_cd8_ds17203), tonumber(row.crp1_fbrain_ds11872), tonumber(row.crp1_fheart_ds12531), tonumber(row.crp1_fintestine_lg_ds17313), tonumber(row.crp1_fkidney_ds20786), tonumber(row.crp1_flung_ds14724), tonumber(row.crp1_fmuscle_leg_ds20239), tonumber(row.crp1_fplacenta_ds20346), tonumber(row.crp1_fskin_fibro_leg_r_quad_ds19943), tonumber(row.crp1_fspinal_cord_ds20351), tonumber(row.crp1_fstomach_ds17878), tonumber(row.crp1_fthymus_ds20341), tonumber(row.crp1_gm06990_ds7748), tonumber(row.crp1_gm12865_ds12436), tonumber(row.crp1_haepic_ds12663),
	    tonumber(row.crp1_hah_ds15192), tonumber(row.crp1_hasp_ds14790), tonumber(row.crp1_hcfaa_ds13480), tonumber(row.crp1_hcf_ds12501), tonumber(row.crp1_hcm_ds12599), tonumber(row.crp1_hcpepic_ds12447), tonumber(row.crp1_heepic_ds12763), tonumber(row.crp1_hepg2_ds7764), tonumber(row.crp1_hesct0_ds11909), tonumber(row.crp1_hff_ds15115), tonumber(row.crp1_hgf_ds11752), tonumber(row.crp1_hipepic_ds12684),
	    tonumber(row.crp1_hmf_ds13368), tonumber(row.crp1_hmvec_dblad_ds13337), tonumber(row.crp1_hmvec_dblneo_ds13242), tonumber(row.crp1_hmvec_dlyneo_ds13150), tonumber(row.crp1_hmvec_lbl_ds13372), tonumber(row.crp1_hmvec_lly_ds13185), tonumber(row.crp1_hpaf_ds13411), tonumber(row.crp1_hpdlf_ds13573), tonumber(row.crp1_hpf_ds13390), tonumber(row.crp1_hrce_ds10666), tonumber(row.crp1_hsmm_ds14426), tonumber(row.crp1_hth17_ds11039),
	    tonumber(row.crp1_hth1_ds7840), tonumber(row.crp1_hth2ds17597), tonumber(row.crp1_htr_ds14702), tonumber(row.crp1_huvec_ds10060), tonumber(row.crp1_hvmf_ds13981), tonumber(row.crp1_imr90_ds13219), tonumber(row.crp1_ips_19_11_ds15153), tonumber(row.crp1_ith1_ds18018), tonumber(row.crp1_ith2_ds17603), tonumber(row.crp1_k562_ds9767), tonumber(row.crp1_lhcn_m2_ds20548), tonumber(row.crp1_m059j_ds20493), tonumber(row.crp1_mesendoderm_ds19310), tonumber(row.crp1_msc_ds21042), tonumber(row.crp1_nb4_ds12543), tonumber(row.crp1_nha_ds12800),
	    tonumber(row.crp1_nhdf_ad_ds12863), tonumber(row.crp1_nhdf_neo_ds11923), tonumber(row.crp1_nhlf_ds12829), tonumber(row.crp1_psoas_muscle_ds20325), tonumber(row.crp1_rpmi_7951_ds20909), tonumber(row.crp1_saec_ds10518), tonumber(row.crp1_skin_fibroblasts_ds18224), tonumber(row.crp1_skin_keratinocytes_ds18692),
	    tonumber(row.crp1_skin_melanocytes_ds18590), tonumber(row.crp1_skmc_ds11949), tonumber(row.crp1_sknsh_ds8482), tonumber(row.crp1_small_bowel_mucosa_ds20770), tonumber(row.crp1_t_47d_ds19794), tonumber(row.crp1_trophoblast_ds19317), tonumber(row.crp1_vhmec_ds18406), tonumber(row.crp2_a549_ds14289), 
	      tonumber(row.crp2_ag10803_ds12374), tonumber(row.crp2_aoaf_ds13513), tonumber(row.crp2_cd14_ds17215), tonumber(row.crp2_cd19_ds17186), tonumber(row.crp2_cd20_ds18208), tonumber(row.crp2_cd34_ds12274), tonumber(row.crp2_cd3_cordblood_ds17706), tonumber(row.crp2_cd3_ds17198), tonumber(row.crp2_cd4_ds17212), tonumber(row.crp2_cd4pos_n_ds14108), tonumber(row.crp2_cd56_ds17189), tonumber(row.crp2_cd8_ds17203),
	      tonumber(row.crp2_fbrain_ds11872), tonumber(row.crp2_fheart_ds12531), tonumber(row.crp2_fintestine_lg_ds17313), tonumber(row.crp2_fkidney_ds20786), tonumber(row.crp2_flung_ds14724), tonumber(row.crp2_fmuscle_leg_ds20239), tonumber(row.crp2_fplacenta_ds20346), tonumber(row.crp2_fskin_fibro_leg_r_quad_ds19943),
	      tonumber(row.crp2_fspinal_cord_ds20351), tonumber(row.crp2_fstomach_ds17878), tonumber(row.crp2_fthymus_ds20341), tonumber(row.crp2_gm06990_ds7748), tonumber(row.crp2_gm12865_ds12436), tonumber(row.crp2_haepic_ds12663), tonumber(row.crp2_hah_ds15192), tonumber(row.crp2_hasp_ds14790), tonumber(row.crp2_hcfaa_ds13480), tonumber(row.crp2_hcf_ds12501), tonumber(row.crp2_hcm_ds12599), tonumber(row.crp2_hcpepic_ds12447),
	      tonumber(row.crp2_heepic_ds12763), tonumber(row.crp2_hepg2_ds7764), tonumber(row.crp2_hesct0_ds11909), tonumber(row.crp2_hff_ds15115), tonumber(row.crp2_hgf_ds11752), tonumber(row.crp2_hipepic_ds12684), tonumber(row.crp2_hmf_ds13368), tonumber(row.crp2_hmvec_dblad_ds13337), tonumber(row.crp2_hmvec_dblneo_ds13242), tonumber(row.crp2_hmvec_dlyneo_ds13150), tonumber(row.crp2_hmvec_lbl_ds13372), tonumber(row.crp2_hmvec_lly_ds13185), tonumber(row.crp2_hpaf_ds13411), tonumber(row.crp2_hpdlf_ds13573), tonumber(row.crp2_hpf_ds13390),
	      tonumber(row.crp2_hrce_ds10666), tonumber(row.crp2_hsmm_ds14426), tonumber(row.crp2_hth17_ds11039), tonumber(row.crp2_hth1_ds7840), tonumber(row.crp2_hth2ds17597), tonumber(row.crp2_htr_ds14702), tonumber(row.crp2_huvec_ds10060), tonumber(row.crp2_hvmf_ds13981), tonumber(row.crp2_imr90_ds13219),
	      tonumber(row.crp2_ips_19_11_ds15153), tonumber(row.crp2_ith1_ds18018), tonumber(row.crp2_ith2_ds17603), tonumber(row.crp2_k562_ds9767), tonumber(row.crp2_lhcn_m2_ds20548), tonumber(row.crp2_m059j_ds20493), tonumber(row.crp2_mesendoderm_ds19310), tonumber(row.crp2_msc_ds21042), tonumber(row.crp2_nb4_ds12543),
	      tonumber(row.crp2_nha_ds12800), tonumber(row.crp2_nhdf_ad_ds12863), tonumber(row.crp2_nhdf_neo_ds11923), tonumber(row.crp2_nhlf_ds12829), tonumber(row.crp2_psoas_muscle_ds20325), tonumber(row.crp2_rpmi_7951_ds20909),
	      tonumber(row.crp2_saec_ds10518), tonumber(row.crp2_skin_fibroblasts_ds18224), tonumber(row.crp2_skin_keratinocytes_ds18692), tonumber(row.crp2_skin_melanocytes_ds18590), tonumber(row.crp2_skmc_ds11949), tonumber(row.crp2_sknsh_ds8482),
	      tonumber(row.crp2_small_bowel_mucosa_ds20770), tonumber(row.crp2_t_47d_ds19794), tonumber(row.crp2_trophoblast_ds19317), tonumber(row.crp2_vhmec_ds18406), tonumber(labelValue) });
   
    
  
  elseif dataSource=="Miriam2014" then
  
    readTensor = torch.Tensor({tonumber(row.name), 
      tonumber(row.crp1_id_region), tonumber(row.crp2_id_region), 
      tonumber(row.crp1_gm12878), 
      tonumber(row.crp1_h1hesc), 
      tonumber(row.crp1_helas3), 
      tonumber(row.crp1_hepg2), 
      tonumber(row.crp1_huvec), 
      tonumber(row.crp1_k562),
      tonumber(row.crp2_gm12878), 
      tonumber(row.crp2_h1hesc), 
      tonumber(row.crp2_helas3), 
      tonumber(row.crp2_hepg2), 
      tonumber(row.crp2_huvec), 
      tonumber(row.crp2_k562),  
      tonumber(labelValue) });        
      
  end
  
  local newReadTensor = torch.Tensor();
  
  if dnaseExcludeColumnNumber>=1 and dnaseExcludeColumnNumber<=(numberOfCellTypes+1) then   
    
    local columnNames = getColumnNamesOfTable("chromregionprofiles")
    
    if printOnce == true then
      print("EXCLUDING THE FEATURE "..dnaseExcludeColumnName.." number "..dnaseExcludeColumnNumber.." among "..(numberOfCellTypes+1));
      printOnce = false
    end
    
    local firstInitIndex = 1
    local firstEndIndex = first_profile_initial+dnaseExcludeColumnNumber-2
    local secondInitIndex = first_profile_initial+dnaseExcludeColumnNumber
    local secondEndIndex = second_profile_initial+dnaseExcludeColumnNumber-1
    local thirdInitIndex = second_profile_initial+dnaseExcludeColumnNumber+1
    local thirdEndIndex = (#(readTensor))[1]
    
--     print("firstInitIndex = "..firstInitIndex)
--     print("firstEndIndex = "..firstEndIndex)
--     print("secondInitIndex = "..secondInitIndex)
--     print("secondEndIndex = "..secondEndIndex)
--     print("thirdInitIndex = "..thirdInitIndex)
--     print("thirdEndIndex = "..thirdEndIndex)
    
    
    newReadTensor = (readTensor)[{{firstInitIndex,firstEndIndex}}]:cat((readTensor)[{{secondInitIndex, secondEndIndex}}]):cat(( readTensor)[{{thirdInitIndex, thirdEndIndex}}])
  else
    newReadTensor = readTensor;
  end
  
  
  -- THE MAXIMUM VALUES BECOME ALL MEAN = 200
  local p=0;

  if PRINT_MANIPULATION==true then 
    print("MAXIMUM VALUES MANIPULATION"); PRINT_MANIPULATION = false; 
  end
  for p=first_profile_initial, second_profile_finish do
     if newReadTensor[p]> MAX_MEAN then newReadTensor[p] = MAX_MEAN end;
  end
  
  return newReadTensor;
end

-- rounds a real number num to the number having idp values after the dot
function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult;
end

-- Function that prints time
function printTime(timeStart, stringToPrint)
	timeEnd = os.time();
	duration = timeEnd - timeStart;
	print('\nduration '..stringToPrint.. ': '.. tonumber(duration).. ' seconds');
	io.flush();
	print('duration '..stringToPrint.. ': '..string.format("%.2d days, %.2d hours, %.2d minutes, %.2d seconds", (duration/(60*60))/24, duration/(60*60)%24, duration/60%60, duration%60)) 
	io.flush();
	
      return duration;
end
   

-- function that returns chrstart and chrend of the specific interaction
function specificInteractionQuery(chromSel, row_number, dataSource, startOrEnd)  
  
  local sql_query = "";
  if startOrEnd == "start" then  sql_query = " SELECT cr1_chrstart ";
  elseif startOrEnd == "end" then  sql_query = " SELECT cr1_chrend "; end
  
  sql_query = sql_query .. " FROM ( "..
  "  SELECT cr1.chrstart AS cr1_chrstart, cr1.chrend AS cr1_chrend, " ..
  "  ROW_NUMBER() OVER (ORDER BY cr1.chrstart ) AS Row_Counter ";
  
  if dataSource=="Miriam2014" then
    
    sql_query = sql_query .. "   FROM hic_interactions_with_labels_and_ids AS i "..
    "   JOIN chromregions  AS cr1 "..
    "   ON (i.id_region1=cr1.id_region) "..
    "   JOIN chromregions AS cr2 "..
    "   ON (i.id_region2=cr2.id_region) ";  
    
   elseif dataSource=="Thurman_Miriam" then
    
    sql_query = sql_query .. " FROM hic_interactions_with_labels_and_ids AS i "..
    " JOIN chromregions AS cr1 "..
    " ON (i.id_region1=cr1.id_region) "..
    " JOIN chromregions AS cr2 "..
    " ON (i.id_region2=cr2.id_region) ";    
    
   elseif dataSource=="Thurman2012" then
     
      sql_query = sql_query .. "   FROM trueinteractions AS i "..
    "   JOIN chromregions AS cr1 "..
    "   ON (i.id_region1=cr1.id_region) "..
    "   JOIN chromregions AS cr2 "..
    "   ON (i.id_region2=cr2.id_region) ";  
  end
  
  sql_query = sql_query .. "   JOIN chromosomes AS c ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr) "..
  "   WHERE c.name='"..chromSel.."' AND cr1.chrstart <> cr1.chrend "..
  " ) q "..
  " WHERE Row_Counter = '"..row_number.."' "..
  " ORDER BY cr1_chrstart, cr1_chrend; ";

  -- print("\tsql_query:"..sql_query);
  -- os.exit();
  
  return sql_query;

end


-- function that divides the genome into N spans having a fixed span size
function selectGenomeSpanIndices_bySpanSize(chromSel, true_interactions_spanSize, dataSource)  
      
  -- retrieve the position of the first locus
  local sql_query_count_interactions = "";
    
 -- let's count the interactions first  
  sql_query_count_interactions = " SELECT COUNT(*)  ";
  
  if dataSource=="Miriam2014" or dataSource=="Thurman_Miriam" then 
    sql_query_count_interactions =  sql_query_count_interactions .. " FROM hic_interactions_with_labels_and_ids AS i  "..
    " JOIN chromregions AS cr1   "..
    " ON (i.id_region1=cr1.id_region)  "..
    " JOIN chromregions AS cr2  "..
    " ON (i.id_region2=cr2.id_region)  ";
  elseif dataSource=="Thurman2012" then 
      sql_query_count_interactions =  sql_query_count_interactions .. " FROM trueinteractions AS i  "..
    " JOIN chromregions AS cr1   "..
    " ON (i.id_region1=cr1.id_region)  "..
    " JOIN chromregions AS cr2  "..
    " ON (i.id_region2=cr2.id_region)  ";
  end
  
  sql_query_count_interactions =  sql_query_count_interactions .. " JOIN chromosomes AS c ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr)  "..
  " WHERE c.name='"..chromSel.."';  ";
  
--   print("\tsql_query_count_interactions = \n\t"..sql_query_count_interactions);
--   os.exit();
--   
   -- retrieve a cursor
  local cur = assert(openGlobalDbConnection():execute(string.format([[%s]], sql_query_count_interactions)));	  

  -- print all rows, the rows will be indexed by field names
  local row = cur:fetch ({}, "a");

  local number_of_interactions = tonumber(row.count);
  cur:close(); -- already closed because all the result set was consumed
  -- closeGlobalDbConnection()
  
  print("number_of_interactions="..comma_value(number_of_interactions));
  
  local numSpans = -1
  local indices_start = {}
  local indices_end = {}
  local indicesTab = {}
  
  -- now let's create the spans
  if number_of_interactions < true_interactions_spanSize then 
    print("ERROR: there are less interactions that the minimum required for the spans\nThe program will exit");
    os.exit();
    
  else    
    numSpans = math.ceil(number_of_interactions/true_interactions_spanSize);    
  end
  
  -- for each span, retrieve the first and the last locus  
  for i=1,numSpans do
    
    indicesTab[i] = i;
    
    local rate = round(i*100/numSpans,2);
    io.write("[rate="..rate.."%] ");
    io.flush();
      
      local row_number = (i*true_interactions_spanSize)-true_interactions_spanSize+1
	-- retrieve the position of the first locus
      sql_query_first_locus_start = specificInteractionQuery(chromSel, row_number, dataSource, "start");
      
        -- retrieve a cursor
	local cur2 = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query_first_locus_start)));	  

	local row2 = cur2:fetch ({}, "a");
	local first_locus_start = tonumber(row2.cr1_chrstart);
	cur2:close();
	
	-- print("\n\tsql_query_first_locus_start: "..sql_query_first_locus_start);
      
      
     if i~=numSpans then row_number = row_number+true_interactions_spanSize-1;
     else row_number = number_of_interactions; end
      
	-- retrieve the position of the last locus
      sql_query_last_locus_end = specificInteractionQuery(chromSel, row_number, dataSource, "end");  
      
       -- print("\n\tsql_query_last_locus_end: "..sql_query_last_locus_end);
      
              -- retrieve a cursor
	local cur3 = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query_last_locus_end)));
		-- print(sql_query_last_locus_end);
	local row3 = cur3:fetch ({}, "a");
	local last_locus_end = tonumber(row3.cr1_chrend);
	cur3:close()
	    
     print("(i="..i..") first_locus_start="..comma_value(first_locus_start).."\tlast_locus_end="..comma_value(last_locus_end));
     
     -- closeGlobalDbConnection()
     
     indices_start[i] = first_locus_start;
     indices_end[i] = last_locus_end;
     
  end 
  
  -- let's permute the indices
  
  local indicesNewTab = permute(indicesTab, numSpans, numSpans);
  local new_indices_start = {};
  local new_indices_end = {};
  
  for i=1, #indicesNewTab do
    
      new_indices_start[i] = indices_start[indicesNewTab[i]]
      new_indices_end[i] = indices_end[indicesNewTab[i]]
    
      io.write("new_indices_start["..i.."]= ".."indices_start["..indicesNewTab[i].."]= "..comma_value(new_indices_start[i]).."\t"); 
      io.flush();
      io.write("new_indices_end["..i.."]= " .."indices_end["..indicesNewTab[i].."]= " ..comma_value(new_indices_end[i]).."\t"); 
      io.flush();
      local diff = indices_end[i] - indices_start[i];
      io.write("size in basepairs: " ..comma_value(diff).."\n");  
    end  
  
  
  return {new_indices_start, new_indices_end};
  
end


-- function that divides the genome into numSpans spans and return the indices
function selectGenomeSpanIndices_byNumSpans(chromSel, numSpans, dataSource)
      
  -- retrieve the position of the first locus
  local sql_query_first_locus = "";
  
  if dataSource=="Thurman2012" or dataSource=="Thurman_Miriam" then 
    sql_query_first_locus = "SELECT cr.chrstart  "..
			  "FROM chromregions AS cr "..
			  "JOIN chromosomes AS c  "..
			  "ON c.id_chr=cr.id_chr  "..
			  "WHERE c.name='"..chromSel.."' "..
			  "ORDER BY cr.chrstart "..
			  "LIMIT 1; "
  elseif dataSource=="Miriam2014" then 
    sql_query_first_locus = "SELECT cr.chrstart  "..
			  "FROM chromregions AS cr ".. -- TO CHECK
			  "JOIN chromosomes AS c  "..
			  "ON c.id_chr=cr.id_chr  "..
			  "WHERE c.name='"..chromSel.."' "..
			  "ORDER BY cr.chrstart "..
			  "LIMIT 1; "
  end
	
  --print("\tsql_query_first_locus: \n"..sql_query_first_locus);
  io.flush();

  -- retrieve a cursor
  local  cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query_first_locus)));	  

  -- print all rows, the rows will be indexed by field names
  local row = cur:fetch ({}, "a");

  local first_locus = tonumber(row.chrstart);
  cur:close(); -- already closed because all the result set was consumed

  print("first_locus= ".. first_locus);
    
      -- retrieve the position of the last locus
  local sql_query_last_locus = "";
  
  if dataSource=="Thurman2012" or dataSource=="Thurman_Miriam" then 
    sql_query_last_locus =  "SELECT cr.chrend  "..
			  "FROM chromregions AS cr "..
			  "JOIN chromosomes AS c  "..
			  "ON c.id_chr=cr.id_chr  "..
			  "WHERE c.name='"..chromSel.."' "..
			  "ORDER BY cr.chrend DESC "..
			  "LIMIT 1; "
  elseif dataSource=="Miriam2014" then 
    sql_query_last_locus =  "SELECT cr.chrend  "..
			  "FROM chromregions AS cr ".. -- TO CHECK
			  "JOIN chromosomes AS c  "..
			  "ON c.id_chr=cr.id_chr  "..
			  "WHERE c.name='"..chromSel.."' "..
			  "ORDER BY cr.chrend DESC "..
			  "LIMIT 1; "
  end
	
   --print("\tsql_query_last_locus: \n"..sql_query_last_locus);
   io.flush();

   -- retrieve a cursor
   local cur2 = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query_last_locus)));
	  

    -- print all rows, the rows will be indexed by field names
   local row2 = cur2:fetch ({}, "a");

   local last_locus = tonumber(row2.chrend);
   print("last_locus= ".. last_locus);
    
    -- close everything

    cur2:close(); -- already closed because all the result set was consumed
    -- closeGlobalDbConnection()
    
    -- ### Compute the indices
    
    local total_genome_length = last_locus - first_locus;
    local first_span_size = round(total_genome_length/numSpans,0);
    
    --sub_span_times = 5 -- the spans for the 2nd to the last have to be 10 times smaller than the first one
    --other_span_size = round(total_genome_length/(numSpans*sub_span_times),0);
    
    
    position_limit = 500000;
   
    print("total_genome_length ".. total_genome_length );
    io.flush();
    print("first_span_size = "..first_span_size);
    io.flush();
--     print("other_span_size = "..other_span_size);
--     io.flush();
    
    totalSpanNumber = numSpans
    print("totalSpanNumber = "..totalSpanNumber);
    io.flush();
    
    indices_start = {}
    indices_symmetric_start = {}
    indices_end = {}
    indices_start[1] = first_locus
    indices_symmetric_start[1] = first_locus
    indices_end[1] = first_locus + first_span_size - 1
    
    -- original
    for i=2, numSpans do
      indices_symmetric_start[i] = indices_symmetric_start[i-1] + first_span_size
      indices_start[i] = indices_symmetric_start[i-1] + first_span_size - position_limit
      indices_end[i] = indices_end[i-1] + first_span_size
    end  
    
--       indices_symmetric_start[2] = indices_end[1]
--       indices_start[2] = indices_symmetric_start[1] + other_span_size - position_limit
--       indices_end[2] = indices_start[2] + other_span_size
-- 
--     for i=3, totalSpanNumber do
--       indices_symmetric_start[i] = indices_symmetric_start[i-1] + other_span_size
--       indices_start[i] = indices_symmetric_start[i-1] + other_span_size - position_limit
--       indices_end[i] = indices_end[i-1] + other_span_size
--     end  
  
     
    -- we force the last index of the last span to be the last_locus
    -- to include the complete genome
    
    -- original
    -- indices_end[numSpans] = last_locus
    
    indices_end[totalSpanNumber] = last_locus
    
    for i=1, totalSpanNumber do
      io.write("indices_start["..i.."]: "..indices_start[i].."\t"); 
      io.flush();
      io.write("indices_end["..i.."]: " ..indices_end[i].."\t"); 
      io.flush();
      diff = indices_end[i] - indices_start[i];
      io.write("size in basepairs: " ..diff.."\n"); 
      io.flush();
      
--       if i>=2 then span = indices_start[i]- indices_end[i-1]
--        io.write(" indices_start["..i.."]- indices_end["..(i-1).."]: " ..span.."\n");
--       end      
    end  
    
    --os.exit();
    
    return {indices_start, indices_end};
end


-- Function get the names of a table (e.g. the cell type names)
function getColumnNamesOfTable(tableName)
  
	--print("getColumnNamesOfTable(tableName) START");
  
	local sql_query = "SELECT * FROM information_schema.columns WHERE table_name   = '"..tableName.."';"
		
	-- print("\n sql_query: "..sql_query)
	local cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query)));	  

	-- print all rows, the rows will be indexed by field names
	local row = cur:fetch ({}, "a");	
	
	local columnNames = {}
	local i = 1
	while row do
	   columnNames[i] = tostring(row.column_name);	  
	   -- print("columnNames["..i.."] "..columnNames[i]);
	   row = cur:fetch(row, "a");
	   i = i + 1
	end
		
	cur:close(); -- already closed because all the result set was consumed
	-- closeGlobalDbConnection()

	--print("getColumnNamesOfTable(tableName) END");
	
	return columnNames;  
end


-- Function that reads a chromosome region ID and returns a chromRegion profile
function fromIdToChromRegion(chromSel, chromRegionId)
  
  if chromSel==nil or chromRegionId==nil then print("fromIdToChromRegion(chromSel, chromRegionId) Error, program will stop "); os.exit(); end
  
        local tmp = chromSel:gsub("chr","");
	local chrNumber = tmp;
	if (tmp=="X") then chrNumber = 23; 
	elseif (tmp=="Y") then chrNumber = 24; 
	else chrNumber = tonumber(tmp);
	end

    -- print("chromRegionId "..chromRegionId);
  
	local sql_query = "SELECT id_chr, id_region,  chrstart, chrend FROM chromregions AS cr WHERE cr.id_region="..tonumber(chromRegionId).." AND cr.id_chr="..tonumber(chrNumber)..";"
	
	
	-- print("\n sql_query: "..sql_query)
	local cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query)));	  

	-- print all rows, the rows will be indexed by field names
	local row = cur:fetch ({}, "a");	
	local chromRegionProfile = {};
	
	while row do
	  chromRegionProfile = {tonumber(row.id_chr), tonumber(row.id_region), tonumber(row.chrstart), tonumber(row.chrend)}	  
	  
	   row = cur:fetch(row, "a");
	end
		
	cur:close(); -- already closed because all the result set was consumed
	-- closeGlobalDbConnection()

	return chromRegionProfile;
end


-- Function that reads a list of interaction ID's and returns the list of chromRegion profiles
function fromInteractionIdsToChromRegions(chromSel, interactionList, values)
   
  if chromSel==nil then print("chromSel==nil fromInteractionIdsToChromRegions() Error, program will stop "); os.exit(); end
  if interactionList==nil then print("interactionList==nil fromInteractionIdsToChromRegions() Error, program will stop "); os.exit(); end
  -- if #interactionList == 0 then print("#interactionList == 0 fromInteractionIdsToChromRegions() Error, program will stop "); os.exit(); end
  -- if #interactionList[1]==0 then print("#interactionList[1]==0 fromInteractionIdsToChromRegions() Error, program will stop "); os.exit(); end
 
  
  
    local chromRegionsList = {};   
    tmp = chromSel:gsub("chr","");
    chrNumber = tmp;
    if (tmp=="X") then chrNumber = 23; 
    elseif (tmp=="Y") then chrNumber = 24;
    else chrNumber = tonumber(tmp);
    end
	
    print("Final list of False Positive (FP) interactions:");
    for i=1,#interactionList do  
      local firstChromRegionId = interactionList[i][1]
      local secondChromRegionId = interactionList[i][2]
      
      local firstChromRegion = fromIdToChromRegion(chromSel, firstChromRegionId)
      local secondChromRegion = fromIdToChromRegion(chromSel, secondChromRegionId)
      
      
      chromRegionsList[i] = {firstChromRegion, secondChromRegion}
      
      local firstChrName = firstChromRegion[1]
      if firstChromRegion[1]==23 then firstChrName = "X" end
      if firstChromRegion[1]==24 then firstChrName = "Y" end
      local secondChrName = secondChromRegion[1]      
      if secondChromRegion[1]==23 then secondChrName = "X" end
      if secondChromRegion[1]==24 then secondChrName = "Y" end
      
      io.write("["..i.."] chr"..firstChrName.."-"..firstChromRegion[3].."-"..firstChromRegion[4]);
      io.write(" and chr"..secondChrName.."-"..secondChromRegion[3].."-"..secondChromRegion[4]);
      io.write(": value "..tonumber(values[i]).."\n");
      
    end
  
   return chromRegionsList;
end


-- Function that creates the Sql query that reads the TRUE interactions from the PostgreSQL database Miriam2014 dataset
function dbMiriam2014profiles_on_Miriam2014hicinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, tuple_limit)
  
  local sql_query = "\nSELECT ".. chrNumber  .." AS name, crp1.id_region AS crp1_id_region, crp2.id_region AS crp2_id_region, crp1.gm12878 AS crp1_gm12878, crp1.h1hesc AS crp1_h1hesc, crp1.helas3 AS crp1_helas3, crp1.hepg2 AS crp1_hepg2, " ..
  " crp1.huvec AS crp1_huvec, crp1.k562 AS crp1_k562, crp2.gm12878 AS crp2_gm12878, crp2.h1hesc AS crp2_h1hesc, crp2.helas3 AS crp2_helas3, crp2.hepg2 AS crp2_hepg2, crp2.huvec AS crp2_huvec, crp2.k562 AS crp2_k562 " ..
  " FROM hic_interactions_with_labels_and_ids AS ti " .. 
  " JOIN encode_chromregionprofiles_new AS crp1 ON  ti.id_region1=crp1.id_region " ..
  " JOIN encode_chromregionprofiles_new AS crp2 ON  ti.id_region2=crp2.id_region  " ..
  " JOIN chromregions AS cr1 ON cr1.id_region=crp1.id_region " ..
  " JOIN chromregions AS cr2 ON cr2.id_region=crp2.id_region " ..
  " JOIN chromosomes AS c1  ON c1.id_chr=cr1.id_chr " ..
  " JOIN chromosomes AS c2  ON c2.id_chr=cr2.id_chr " .. 
  " WHERE c1.name='"..chromSel.. "'  AND c2.name='"..chromSel.. "' AND  " ..
  " cr1.chrstart>="..tonumber(chrStart_locus).." AND  " ..  
  " cr1.chrend<"..tonumber(chrEnd_locus).." AND  " .. 
  " cr2.chrstart>="..tonumber(chrStart_locus).." AND  " ..
  " cr2.chrend<"..tonumber(chrEnd_locus).." AND crp1.id_region <> crp2.id_region  "  .. " ORDER BY random() "
  
    -- " ORDER BY crp1_id_region, crp2_id_region "  
  
  if tuple_limit ~= -1 and tuple_limit ~= "-1" then sql_query = sql_query .. " LIMIT "..tuple_limit; end
	
  sql_query = sql_query .. ";"  
  -- print("\n\n\n"..sql_query.."\n\n\n");
  
  return sql_query;
  
end


-- Function that creates the upper part of the query to read the Thurman2012 
-- chromosome region profiles
function dbThurman2012_column_selection_query_subpart(chrNumber)
  
  local sql_query = "\nSELECT ";
  
  if chrNumber ~= 0 then
    sql_query = sql_query .. " ".. chrNumber  .." AS name, "
  else
    sql_query = sql_query .. " cr1.id_chr AS name, "
  end
  
  sql_query = sql_query .. " crp1.id_region AS crp1_id_region, crp2.id_region AS crp2_id_region, crp1.a549_ds14289 AS crp1_a549_ds14289,  crp1.ag10803_ds12374 AS crp1_ag10803_ds12374,  crp1.aoaf_ds13513 AS crp1_aoaf_ds13513,  crp1.cd14_ds17215 AS crp1_cd14_ds17215,  crp1.cd19_ds17186 AS crp1_cd19_ds17186,  crp1.cd20_ds18208 AS crp1_cd20_ds18208,  crp1.cd34_ds12274 AS crp1_cd34_ds12274,  crp1.cd3_cordblood_ds17706 AS crp1_cd3_cordblood_ds17706,  crp1.cd3_ds17198 AS crp1_cd3_ds17198,  crp1.cd4_ds17212 AS crp1_cd4_ds17212,  crp1.cd4pos_n_ds14108 AS crp1_cd4pos_n_ds14108,  crp1.cd56_ds17189 AS crp1_cd56_ds17189,  crp1.cd8_ds17203 AS crp1_cd8_ds17203,  crp1.fbrain_ds11872 AS crp1_fbrain_ds11872,  crp1.fheart_ds12531 AS crp1_fheart_ds12531,  crp1.fintestine_lg_ds17313 AS crp1_fintestine_lg_ds17313,  crp1.fkidney_ds20786 AS crp1_fkidney_ds20786,  crp1.flung_ds14724 AS crp1_flung_ds14724,  crp1.fmuscle_leg_ds20239 AS crp1_fmuscle_leg_ds20239,  crp1.fplacenta_ds20346 AS crp1_fplacenta_ds20346,  crp1.fskin_fibro_leg_r_quad_ds19943 AS crp1_fskin_fibro_leg_r_quad_ds19943,  crp1.fspinal_cord_ds20351 AS crp1_fspinal_cord_ds20351,  crp1.fstomach_ds17878 AS crp1_fstomach_ds17878,  crp1.fthymus_ds20341 AS crp1_fthymus_ds20341,  crp1.gm06990_ds7748 AS crp1_gm06990_ds7748,  crp1.gm12865_ds12436 AS crp1_gm12865_ds12436,  crp1.haepic_ds12663 AS crp1_haepic_ds12663,  crp1.hah_ds15192 AS crp1_hah_ds15192,  crp1.hasp_ds14790 AS crp1_hasp_ds14790,  crp1.hcfaa_ds13480 AS crp1_hcfaa_ds13480,  crp1.hcf_ds12501 AS crp1_hcf_ds12501,  crp1.hcm_ds12599 AS crp1_hcm_ds12599,  crp1.hcpepic_ds12447 AS crp1_hcpepic_ds12447,  crp1.heepic_ds12763 AS crp1_heepic_ds12763,  crp1.hepg2_ds7764 AS crp1_hepg2_ds7764,  crp1.hesct0_ds11909 AS crp1_hesct0_ds11909,  crp1.hff_ds15115 AS crp1_hff_ds15115, ";
	
	sql_query = sql_query .. " crp1.hgf_ds11752 AS crp1_hgf_ds11752,  crp1.hipepic_ds12684 AS crp1_hipepic_ds12684,  crp1.hmf_ds13368 AS crp1_hmf_ds13368,  crp1.hmvec_dblad_ds13337 AS crp1_hmvec_dblad_ds13337,  crp1.hmvec_dblneo_ds13242 AS crp1_hmvec_dblneo_ds13242,  crp1.hmvec_dlyneo_ds13150 AS crp1_hmvec_dlyneo_ds13150,  crp1.hmvec_lbl_ds13372 AS crp1_hmvec_lbl_ds13372,  crp1.hmvec_lly_ds13185 AS crp1_hmvec_lly_ds13185,  crp1.hpaf_ds13411 AS crp1_hpaf_ds13411,  crp1.hpdlf_ds13573 AS crp1_hpdlf_ds13573,  crp1.hpf_ds13390 AS crp1_hpf_ds13390,  crp1.hrce_ds10666 AS crp1_hrce_ds10666,  crp1.hsmm_ds14426 AS crp1_hsmm_ds14426,  crp1.hth17_ds11039 AS crp1_hth17_ds11039,  crp1.hth1_ds7840 AS crp1_hth1_ds7840,  crp1.hth2ds17597 AS crp1_hth2ds17597,  crp1.htr_ds14702 AS crp1_htr_ds14702,  crp1.huvec_ds10060 AS crp1_huvec_ds10060,  crp1.hvmf_ds13981 AS crp1_hvmf_ds13981,  crp1.imr90_ds13219 AS crp1_imr90_ds13219,  crp1.ips_19_11_ds15153 AS crp1_ips_19_11_ds15153,  crp1.ith1_ds18018 AS crp1_ith1_ds18018,  crp1.ith2_ds17603 AS crp1_ith2_ds17603,  crp1.k562_ds9767 AS crp1_k562_ds9767,  crp1.lhcn_m2_ds20548 AS crp1_lhcn_m2_ds20548,  crp1.m059j_ds20493 AS crp1_m059j_ds20493, "
	
	sql_query = sql_query .. " crp1.mesendoderm_ds19310 AS crp1_mesendoderm_ds19310,  crp1.msc_ds21042 AS crp1_msc_ds21042,  crp1.nb4_ds12543 AS crp1_nb4_ds12543,  crp1.nha_ds12800 AS crp1_nha_ds12800,  crp1.nhdf_ad_ds12863 AS crp1_nhdf_ad_ds12863,  crp1.nhdf_neo_ds11923 AS crp1_nhdf_neo_ds11923,  crp1.nhlf_ds12829 AS crp1_nhlf_ds12829,  crp1.psoas_muscle_ds20325 AS crp1_psoas_muscle_ds20325,  crp1.rpmi_7951_ds20909 AS crp1_rpmi_7951_ds20909,  crp1.saec_ds10518 AS crp1_saec_ds10518,  crp1.skin_fibroblasts_ds18224 AS crp1_skin_fibroblasts_ds18224,  crp1.skin_keratinocytes_ds18692 AS crp1_skin_keratinocytes_ds18692,  crp1.skin_melanocytes_ds18590 AS crp1_skin_melanocytes_ds18590,  crp1.skmc_ds11949 AS crp1_skmc_ds11949,  crp1.sknsh_ds8482 AS crp1_sknsh_ds8482,  crp1.small_bowel_mucosa_ds20770 AS crp1_small_bowel_mucosa_ds20770,  crp1.t_47d_ds19794 AS crp1_t_47d_ds19794,  crp1.trophoblast_ds19317 AS crp1_trophoblast_ds19317,  crp1.vhmec_ds18406 AS crp1_vhmec_ds18406,  crp2.a549_ds14289 AS crp2_a549_ds14289,  crp2.ag10803_ds12374 AS crp2_ag10803_ds12374,  crp2.aoaf_ds13513 AS crp2_aoaf_ds13513,  crp2.cd14_ds17215 AS crp2_cd14_ds17215,  crp2.cd19_ds17186 ";

	sql_query = sql_query .. " AS crp2_cd19_ds17186,  crp2.cd20_ds18208 AS crp2_cd20_ds18208,  crp2.cd34_ds12274 AS crp2_cd34_ds12274,  crp2.cd3_cordblood_ds17706 AS crp2_cd3_cordblood_ds17706,  crp2.cd3_ds17198 AS crp2_cd3_ds17198,  crp2.cd4_ds17212 AS crp2_cd4_ds17212,  crp2.cd4pos_n_ds14108 AS crp2_cd4pos_n_ds14108,  crp2.cd56_ds17189 AS crp2_cd56_ds17189,  crp2.cd8_ds17203 AS crp2_cd8_ds17203,  crp2.fbrain_ds11872 AS crp2_fbrain_ds11872,  crp2.fheart_ds12531 AS crp2_fheart_ds12531,  crp2.fintestine_lg_ds17313 AS crp2_fintestine_lg_ds17313,  crp2.fkidney_ds20786 AS crp2_fkidney_ds20786,  crp2.flung_ds14724 AS crp2_flung_ds14724,  crp2.fmuscle_leg_ds20239 AS crp2_fmuscle_leg_ds20239,  crp2.fplacenta_ds20346 AS crp2_fplacenta_ds20346,  crp2.fskin_fibro_leg_r_quad_ds19943 AS crp2_fskin_fibro_leg_r_quad_ds19943,  crp2.fspinal_cord_ds20351 AS crp2_fspinal_cord_ds20351,  crp2.fstomach_ds17878 AS crp2_fstomach_ds17878,  crp2.fthymus_ds20341 AS crp2_fthymus_ds20341,  crp2.gm06990_ds7748 AS crp2_gm06990_ds7748,  crp2.gm12865_ds12436 AS crp2_gm12865_ds12436,  crp2.haepic_ds12663 AS crp2_haepic_ds12663,  crp2.hah_ds15192 AS crp2_hah_ds15192,  crp2.hasp_ds14790 AS crp2_hasp_ds14790,  crp2.hcfaa_ds13480 AS crp2_hcfaa_ds13480,  crp2.hcf_ds12501 AS crp2_hcf_ds12501,  crp2.hcm_ds12599 AS crp2_hcm_ds12599,  crp2.hcpepic_ds12447 AS crp2_hcpepic_ds12447,  crp2.heepic_ds12763 AS crp2_heepic_ds12763,  crp2.hepg2_ds7764 AS crp2_hepg2_ds7764,  crp2.hesct0_ds11909 AS crp2_hesct0_ds11909,  crp2.hff_ds15115 AS crp2_hff_ds15115,  crp2.hgf_ds11752 AS crp2_hgf_ds11752,  crp2.hipepic_ds12684 AS crp2_hipepic_ds12684,  crp2.hmf_ds13368 AS crp2_hmf_ds13368,  crp2.hmvec_dblad_ds13337 AS crp2_hmvec_dblad_ds13337,  crp2.hmvec_dblneo_ds13242 AS crp2_hmvec_dblneo_ds13242,  crp2.hmvec_dlyneo_ds13150 AS crp2_hmvec_dlyneo_ds13150,  crp2.hmvec_lbl_ds13372 AS crp2_hmvec_lbl_ds13372,  crp2.hmvec_lly_ds13185 AS crp2_hmvec_lly_ds13185,  crp2.hpaf_ds13411 AS crp2_hpaf_ds13411,  crp2.hpdlf_ds13573 AS crp2_hpdlf_ds13573,  crp2.hpf_ds13390 AS crp2_hpf_ds13390,  crp2.hrce_ds10666 AS crp2_hrce_ds10666,  crp2.hsmm_ds14426 AS crp2_hsmm_ds14426,  crp2.hth17_ds11039 AS crp2_hth17_ds11039,  crp2.hth1_ds7840 AS crp2_hth1_ds7840,  crp2.hth2ds17597 AS crp2_hth2ds17597,  crp2.htr_ds14702 AS crp2_htr_ds14702,  crp2.huvec_ds10060 AS crp2_huvec_ds10060,  crp2.hvmf_ds13981 AS crp2_hvmf_ds13981,  crp2.imr90_ds13219 AS crp2_imr90_ds13219,  crp2.ips_19_11_ds15153 AS crp2_ips_19_11_ds15153,  crp2.ith1_ds18018 AS crp2_ith1_ds18018,  crp2.ith2_ds17603 AS crp2_ith2_ds17603,  crp2.k562_ds9767 AS crp2_k562_ds9767, "
	sql_query = sql_query .. " crp2.lhcn_m2_ds20548 AS crp2_lhcn_m2_ds20548,  crp2.m059j_ds20493 AS crp2_m059j_ds20493,  crp2.mesendoderm_ds19310 AS crp2_mesendoderm_ds19310,  crp2.msc_ds21042 AS crp2_msc_ds21042,  crp2.nb4_ds12543 AS crp2_nb4_ds12543,  crp2.nha_ds12800 AS crp2_nha_ds12800,  crp2.nhdf_ad_ds12863 AS crp2_nhdf_ad_ds12863,  crp2.nhdf_neo_ds11923 AS crp2_nhdf_neo_ds11923,  crp2.nhlf_ds12829 AS crp2_nhlf_ds12829,  crp2.psoas_muscle_ds20325 AS crp2_psoas_muscle_ds20325,  crp2.rpmi_7951_ds20909 AS crp2_rpmi_7951_ds20909,  crp2.saec_ds10518 AS crp2_saec_ds10518,  crp2.skin_fibroblasts_ds18224 AS crp2_skin_fibroblasts_ds18224,  crp2.skin_keratinocytes_ds18692 AS crp2_skin_keratinocytes_ds18692,  crp2.skin_melanocytes_ds18590 AS crp2_skin_melanocytes_ds18590,  crp2.skmc_ds11949 AS crp2_skmc_ds11949,  crp2.sknsh_ds8482 AS crp2_sknsh_ds8482,  crp2.small_bowel_mucosa_ds20770 AS crp2_small_bowel_mucosa_ds20770,  crp2.t_47d_ds19794 AS crp2_t_47d_ds19794,  crp2.trophoblast_ds19317 AS crp2_trophoblast_ds19317,  crp2.vhmec_ds18406  AS crp2_vhmec_ds18406 " ;
  
    return sql_query;
  
end



-- Function that creates the Sql query that reads the TRUE interactions from the PostgreSQL database Thurman2012 dataset
function dbThurman2012profiles_on_Thurman2012trueinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, tuple_limit)
  
  -- READIN' THE TRUE TUPLES
	local sql_query = dbThurman2012_column_selection_query_subpart(chrNumber);
	
        sql_query = sql_query .." FROM trueinteractions AS ti "..
	" JOIN chromregionprofiles AS crp1  ON ti.id_region1=crp1.id_region     "..
	" JOIN chromregionprofiles AS crp2  ON ti.id_region2=crp2.id_region     "..
	" JOIN chromregions AS cr1 ON cr1.id_region=crp1.id_region     "..
	" JOIN chromregions AS cr2 ON cr2.id_region=crp2.id_region     "..
	" JOIN chromosomes AS c1  ON c1.id_chr=cr1.id_chr   "..
	" JOIN chromosomes AS c2  ON c2.id_chr=cr2.id_chr   "..
	" WHERE c1.name='"..chromSel.. "' AND c2.name='"..chromSel.. "'    "..
	" AND cr1.chrstart>="..tonumber(chrStart_locus).." AND cr1.chrend<"..tonumber(chrEnd_locus).."    "..
	" AND cr2.chrstart>="..tonumber(chrStart_locus).." AND cr2.chrend<"..tonumber(chrEnd_locus)..
	" AND crp1.id_region <> crp2.id_region " .." ORDER BY random() "
	-- " ORDER BY crp1_id_region, crp2_id_region "
	
	if tuple_limit ~= -1 and tuple_limit ~= "-1" then sql_query = sql_query .. " LIMIT "..tuple_limit; end
	
	sql_query = sql_query .. ";"
  
	
	-- print(sql_query); os.exit();
  
      return sql_query;
end


-- Function that creates the Sql query that reads the Thurman2012 chromosome profiles of the hic interactions from the PostgreSQL database Miriam2014 dataset
-- ### USING THIS ###
function dbThurman2012profiles_on_Miriam2014hicinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, tuple_limit, hicCellTypeToExclude, hicCellTypeToConsider)
  
  print("dbThurman2012profiles_on_Miriam2014hicinteractions_query() ")
  if(tonumber(hicCellTypeToExclude)~=-1) then print("EXCLUDING "..hicCellTypeToExclude .." cell type") end
  if(tonumber(hicCellTypeToConsider)~=-1) then print("CONSIDERING only ".. hicCellTypeToConsider .." cell type") end
  
  local sql_query = dbThurman2012_column_selection_query_subpart(chrNumber);

  sql_query = sql_query .."  FROM hic_interactions_with_labels_and_ids AS hic_inte "..
  -- we retrieve the hic start and end of the 1st chrom region
  " JOIN chromregions AS cr1 "..
  " ON hic_inte.id_region1=cr1.id_region "..

  -- we retrieve the hic start and end of the 2nd chrom region
  " JOIN chromregions AS cr2 "..
  " ON hic_inte.id_region2=cr2.id_region "..

--   -- we retrieve the start and end of the 1st chrom region
--   " JOIN chromregions AS cr1 "..
--   " ON (cr1.chrstart = hic_cr1.chrstart "..
--   " AND cr1.chrend = hic_cr1.chrend "..
--   " AND cr1.id_chr = hic_cr1.id_chr) "..
-- 
--   -- we retrieve the start and end of the 1st chrom region
--   " JOIN chromregions AS cr2 "..
--   " ON (cr2.chrstart = hic_cr2.chrstart "..
--   " AND cr2.chrend = hic_cr2.chrend "..
--   " AND cr2.id_chr = hic_cr2.id_chr) "..

  -- we retrieve the Thurman2012 signal profile of the chrom regions
  " JOIN chromregionprofiles AS crp1 "..
  " ON cr1.id_region=crp1.id_region "..
  " JOIN chromregionprofiles AS crp2 "..
  " ON cr2.id_region=crp2.id_region "..
  
  " JOIN chromosomes AS c "..
  " ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr) WHERE ";

  if chromSel~= "chr0" then sql_query = sql_query .." c.name='"..chromSel.. "' AND "; end
  
--   if (tonumber(hicCellTypeToExclude)==-1  and tonumber(hicCellTypeToConsider)==-1) then
--     print("Error: select a cell type to exclude OR to consider for the Hi-C dataset. The program will stop");
--     os.exit();
--   end
  
  if hicCellTypeToExclude~=-1 and hicCellTypeToExclude~="-1" then
    sql_query = sql_query .. " celltype <> '".. hicCellTypeToExclude .."' AND "
  end
  if hicCellTypeToConsider~=-1 and hicCellTypeToConsider~="-1" then
    sql_query = sql_query .. " celltype = '".. hicCellTypeToConsider .."' AND "
  end  
  
  sql_query = sql_query .." cr1.chrstart>="..tonumber(chrStart_locus).." AND cr1.chrend<"..tonumber(chrEnd_locus).."    "..
  " AND cr2.chrstart>="..tonumber(chrStart_locus).." AND cr2.chrend<"..tonumber(chrEnd_locus)..
  " AND crp1.id_region <> crp2.id_region " .. " ORDER BY random() "   
  -- " ORDER BY crp1_id_region, crp2_id_region "

  if tuple_limit ~= -1 and tuple_limit ~= "-1" then sql_query = sql_query .. " LIMIT "..tuple_limit; end

  sql_query = sql_query .. ";"    

--   print("\n\n\n"..sql_query.."\n\n\n");
--   os.exit();
    
  return sql_query;
  
end

-- Function that creates the Sql query that reads the Thurman2012 chromosome profiles of the hic interactions from the PostgreSQL database Miriam2014 dataset, shared by 2, 3, 4 cell types if requested
function dbThurman2012profiles_on_Miriam2014hicinteractions_query_4_cell_types(chrNumber, chromSel, chrStart_locus, chrEnd_locus, tuple_limit, hicCellTypeToExclude1, hicCellTypeToExclude2, hicCellTypeToExclude3, hicCellTypeToExclude4, hicCellTypeToConsider1,  hicCellTypeToConsider2,  hicCellTypeToConsider3,  hicCellTypeToConsider4 )
  
  print("dbThurman2012profiles_on_Miriam2014hicinteractions_query_4_cell_types() ")
  if(tonumber(hicCellTypeToExclude1)~=-1) then print("EXCLUDING "..hicCellTypeToExclude1 .." cell type") end
  if(tonumber(hicCellTypeToExclude2)~=-1) then print("EXCLUDING "..hicCellTypeToExclude2 .." cell type") end
  if(tonumber(hicCellTypeToExclude3)~=-1) then print("EXCLUDING "..hicCellTypeToExclude3 .." cell type") end
  if(tonumber(hicCellTypeToExclude4)~=-1) then print("EXCLUDING "..hicCellTypeToExclude4 .." cell type") end
  if(tonumber(hicCellTypeToConsider1)~=-1) then print("CONSIDERING ".. hicCellTypeToConsider1 .." cell type") end
  if(tonumber(hicCellTypeToConsider2)~=-1) then print("CONSIDERING ".. hicCellTypeToConsider2 .." cell type") end
  if(tonumber(hicCellTypeToConsider3)~=-1) then print("CONSIDERING ".. hicCellTypeToConsider3 .." cell type") end
  if(tonumber(hicCellTypeToConsider4)~=-1) then print("CONSIDERING ".. hicCellTypeToConsider4 .." cell type") end
  
  local sql_query = dbThurman2012_column_selection_query_subpart(chrNumber);

  sql_query = sql_query .."  FROM hic_interactions_with_labels_and_ids AS hic_inte "..
  -- we retrieve the hic start and end of the 1st chrom region
  " JOIN chromregions AS cr1 "..
  " ON hic_inte.id_region1=cr1.id_region "..

  -- we retrieve the hic start and end of the 2nd chrom region
  " JOIN chromregions AS cr2 "..
  " ON hic_inte.id_region2=cr2.id_region "..

  " JOIN chromregionprofiles AS crp1 "..
  " ON cr1.id_region=crp1.id_region "..
  " JOIN chromregionprofiles AS crp2 "..
  " ON cr2.id_region=crp2.id_region "..
  
  " JOIN chromosomes AS c "..
  " ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr) WHERE ";

  if chromSel~= "chr0" then sql_query = sql_query .." c.name='"..chromSel.. "' AND "; end
  
--   if (tonumber(hicCellTypeToExclude)==-1  and tonumber(hicCellTypeToConsider)==-1) then
--     print("Error: select a cell type to exclude OR to consider for the Hi-C dataset. The program will stop");
--     os.exit();
--   end

   -- 1st cell type to include
  if hicCellTypeToConsider1~=-1 and hicCellTypeToConsider1~="-1" then
    sql_query = sql_query .. " celltype = '".. hicCellTypeToConsider1 .."' AND "
  end  
  
  sql_query = sql_query .." cr1.chrstart>="..tonumber(chrStart_locus).." AND cr1.chrend<"..tonumber(chrEnd_locus).."    "..
  " AND cr2.chrstart>="..tonumber(chrStart_locus).." AND cr2.chrend<"..tonumber(chrEnd_locus)..
  " AND crp1.id_region <> crp2.id_region ";
  
  -- 2nd cell type to include
  if hicCellTypeToConsider2~=-1 and hicCellTypeToConsider2~="-1" then
      sql_query = sql_query .." AND EXISTS ( "..
      " SELECT 1 "..
      " FROM hic_interactions_with_labels_and_ids AS hic_inte  "..
      " JOIN chromregions AS cr1b ON hic_inte.id_region1=cr1b.id_region  "..
      " JOIN chromregions AS cr2b ON hic_inte.id_region2=cr2b.id_region  "..
      " JOIN chromregionprofiles AS crp1b  ON cr1b.id_region=crp1b.id_region  "..
      " JOIN chromregionprofiles AS crp2b  ON cr2b.id_region=crp2b.id_region  "..
      " JOIN chromosomes AS c_bis ON (c_bis.id_chr=cr1b.id_chr AND c_bis.id_chr=cr2b.id_chr) "..
      " WHERE c_bis.id_chr=cr1.id_chr AND  "..
      " celltype = '"..hicCellTypeToConsider2.."'  AND  "..
      " ((crp1.id_region = cr1b.id_region AND "..
      " crp2.id_region = cr2b.id_region) OR "..
      " (crp2.id_region = cr1b.id_region AND "..
      " crp1.id_region = cr2b.id_region) ) AND "..
      " crp1b.id_region <> crp2b.id_region "..
      " ) "
  end
  
  -- 3rd cell type to include
  if hicCellTypeToConsider3~=-1 and hicCellTypeToConsider3~="-1" then
      sql_query = sql_query .." AND EXISTS ( "..
        " SELECT 1 "..
	" FROM hic_interactions_with_labels_and_ids AS hic_inte   "..
	" JOIN chromregions AS cr1c ON hic_inte.id_region1=cr1c.id_region   "..
	" JOIN chromregions AS cr2c ON hic_inte.id_region2=cr2c.id_region   "..
	" JOIN chromregionprofiles AS crp1c  ON cr1c.id_region=crp1c.id_region  ".. 
	" JOIN chromregionprofiles AS crp2c  ON cr2c.id_region=crp2c.id_region   "..
	" JOIN chromosomes AS c_ter ON (c_ter.id_chr=cr1c.id_chr AND c_ter.id_chr=cr2c.id_chr)  "..
	" WHERE c_ter.id_chr=cr1.id_chr AND   "..
	" celltype = '"..hicCellTypeToConsider3.."'  AND   "..
	" ((crp1.id_region = cr1c.id_region AND "..
	" crp2.id_region = cr2c.id_region) OR  "..
	" (crp2.id_region = cr1c.id_region AND "..
	" crp1.id_region = cr2c.id_region) ) AND "..
	" crp1c.id_region <> crp2c.id_region "..
	" ) "
  end
  
  -- 4th cell type to include
  if hicCellTypeToConsider4~=-1 and hicCellTypeToConsider4~="-1" then
      sql_query = sql_query .." AND EXISTS ( "..
	" SELECT 1 "..
	" FROM hic_interactions_with_labels_and_ids AS hic_inte   "..
	" JOIN chromregions AS cr1d ON hic_inte.id_region1=cr1d.id_region   "..
	" JOIN chromregions AS cr2d ON hic_inte.id_region2=cr2d.id_region   "..
	" JOIN chromregionprofiles AS crp1d  ON cr1d.id_region=crp1d.id_region   "..
	" JOIN chromregionprofiles AS crp2d  ON cr2d.id_region=crp2d.id_region   "..
	" JOIN chromosomes AS c_quater ON (c_quater.id_chr=cr1d.id_chr AND  "..
	" c_quater.id_chr=cr2d.id_chr)  "..
	" WHERE c_quater.id_chr=cr1.id_chr AND   "..
	" celltype = '"..hicCellTypeToConsider4.."'  AND   "..
	" ((crp1.id_region = cr1d.id_region AND "..
	" crp2.id_region = cr2d.id_region) OR  "..
	" (crp2.id_region = cr1d.id_region AND "..
	" crp1.id_region = cr2d.id_region) ) AND "..
	" crp1d.id_region <> crp2d.id_region "..
	" ) "
  end
  
--   -- All the 1,2,3,4 cell type to exclude
--   if hicCellTypeToExclude1~=-1 and hicCellTypeToExclude1~="-1" then
--       sql_query = sql_query .." AND NOT EXISTS ( "..
--       " SELECT 1 "..
--       " FROM hic_interactions_with_labels_and_ids AS hic_inte  "..
--       " JOIN chromregions AS cr1e ON hic_inte.id_region1=cr1e.id_region  "..
--       " JOIN chromregions AS cr2e ON hic_inte.id_region2=cr2e.id_region  "..
--       " JOIN chromregionprofiles AS crp1e  ON cr1e.id_region=crp1e.id_region  "..
--       " JOIN chromregionprofiles AS crp2e  ON cr2e.id_region=crp2e.id_region  "..
--       " JOIN chromosomes AS c_five ON (c_five.id_chr=cr1e.id_chr AND c_five.id_chr=cr2e.id_chr) "..
--       " WHERE c_five.id_chr=cr1.id_chr AND  "..
--       " (celltype = '"..hicCellTypeToExclude1.."' OR  "..
--       " celltype = '"..hicCellTypeToExclude2.."' OR  "..
--       " celltype = '"..hicCellTypeToExclude3.."' OR  "..
--       " celltype = '"..hicCellTypeToExclude4.."')  "..
--       " AND ((crp1.id_region = cr1e.id_region AND "..
--       " crp2.id_region = cr2e.id_region) OR "..
--       " (crp2.id_region = cr1e.id_region AND "..
--       " crp1.id_region = cr2e.id_region) ) AND "..
--       " crp1e.id_region <> crp2e.id_region "..
--       " ) "
--   end

  sql_query = sql_query.. " ORDER BY random() ";
  if tuple_limit ~= -1 and tuple_limit ~= "-1" then sql_query = sql_query .. " LIMIT "..tuple_limit; end

  sql_query = sql_query .. ";"    

  -- print("\n\n\n"..sql_query.."\n\n\n");
  -- os.exit();
    
  return sql_query;
  
end


-- Function that creates the Sql query that reads the FALSE interactions from the PostgreSQL database
function dbMiriam2014profiles_on_Miriam2014falseinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, original_tuple_limit)
 
  local sql_query = " SELECT ".. chrNumber  .." AS name, crp1.id_region AS crp1_id_region, crp2.id_region AS crp2_id_region, crp1.gm12878 AS crp1_gm12878, crp1.h1hesc AS crp1_h1hesc, crp1.helas3 AS crp1_helas3, crp1.hepg2 AS crp1_hepg2, "..
  " crp1.huvec AS crp1_huvec, crp1.k562 AS crp1_k562,  crp2.gm12878 AS crp2_gm12878, crp2.h1hesc AS crp2_h1hesc, crp2.helas3 AS crp2_helas3, crp2.hepg2 AS crp2_hepg2, crp2.huvec AS crp2_huvec, crp2.k562 AS crp2_k562 " ..
  " FROM chromregions AS cr1 CROSS JOIN Chromregions AS cr2  " ..
  " JOIN chromosomes AS c  ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr) " .. 
  " JOIN encode_chromregionprofiles_new AS crp1  ON crp1.id_region = cr1.id_region  " ..
  " JOIN encode_chromregionprofiles_new AS crp2  ON crp2.id_region = cr2.id_region  " ..
  " WHERE cr1.id_region <> cr2.id_region  AND NOT EXISTS (    " ..
  "   SELECT 1    " ..
  "   FROM hic_interactions_with_labels_and_ids AS tix    " ..
  "   WHERE cr1.id_region=tix.id_region1    AND cr2.id_region=tix.id_region2   )  ";
  
  if chromSel~= "chr0"  then sql_query = sql_query .. "AND c.name='".. chromSel  .."' "; end
  
  sql_query = sql_query .." AND (cr1.chrend - cr2.chrstart <"..locus_position_limit.. ") " ..
  " AND (cr2.chrend - cr1.chrstart <"..locus_position_limit.. ")  " ..
  " AND cr1.chrstart>="..tonumber(chrStart_locus)..
  " AND cr1.chrend<"..tonumber(chrEnd_locus)..
  " AND cr2.chrstart>="..tonumber(chrStart_locus).. 
  " AND cr2.chrend<"..tonumber(chrEnd_locus) .." ORDER BY random() "
  -- " ORDER BY crp1_id_region, crp2_id_region  "
  
  -- if balancedFlag==true and
  
  if lengthTrues~=0 and lengthTrues~=-1 then sql_query = sql_query ..	 " LIMIT "..lengthTrues 
   elseif original_tuple_limit ~= -1 then sql_query = sql_query ..	 " LIMIT "..original_tuple_limit..";\n" 

   end

	sql_query = sql_query .. ";"

	
	return sql_query;  
  
end

-- Function that creates the Sql query that reads the FALSE interactions from the PostgreSQL database
function dbThurman2012profiles_on_Thurman2012falseinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, original_tuple_limit)
 
  
  local sql_query = dbThurman2012_column_selection_query_subpart(chrNumber);
  
	sql_query = sql_query .. " FROM chromregions AS cr1 " ..
	 " CROSS JOIN chromregions AS cr2 " ..
	 " JOIN chromosomes AS c " ..
	 " ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr) " ..
	 " JOIN chromregionprofiles AS crp1 " ..
	 " ON crp1.id_region = cr1.id_region " ..
	 " JOIN chromregionprofiles AS crp2 " ..
	 " ON crp2.id_region = cr2.id_region " ..
	 " WHERE cr1.id_region <> cr2.id_region " ..
	 " AND NOT EXISTS ( " ..
	 "   SELECT 1 " ..
	 "   FROM trueinteractions AS tix " ..
	 "   WHERE cr1.id_region=tix.id_region1 " ..
	 "   AND cr2.id_region=tix.id_region2 " ..
	 " ) " ..
	 " AND c.name='"..chromSel.. "'" ..
	 " AND (cr1.chrend - cr2.chrstart <"..locus_position_limit.. ") " ..
	 " AND (cr2.chrend - cr1.chrstart <"..locus_position_limit.. ") " ..
	" AND cr1.chrstart>="..tonumber(chrStart_locus)..
	" AND cr1.chrend<"..tonumber(chrEnd_locus)..
	" AND cr2.chrstart>="..tonumber(chrStart_locus)..
	" AND cr2.chrend<"..tonumber(chrEnd_locus)  .." ORDER BY random() "
	-- " ORDER BY crp1_id_region, crp2_id_region "
	
	if balancedFlag==true and lengthTrues~=0 and lengthTrues~=-1 then sql_query = sql_query ..	 " LIMIT "..lengthTrues 
	elseif original_tuple_limit ~= -1 then sql_query = sql_query ..	 " LIMIT "..original_tuple_limit..";\n" 

	end

	
	-- if balancedFlag==false then sql_query = sql_query .. " LIMIT "..tonumber(lengthTrues*2) end -- JUST FOR TRYIN' @@@
	
	sql_query = sql_query .. ";"

	--print("\t\nsql_query = \n" ..sql_query);
	--os.exit();
	
	return sql_query;  
end

-- Function that creates the Sql query that reads the Thurman2012 profiles of the  Miriam2014 FALSE interactions from the PostgreSQL database
-- ### USING THIS ###
function dbThurman2012profiles_on_Miriam2014falseinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, original_tuple_limit, hicCellTypeToExclude, hicCellTypeToConsider)
  
  io.write("dbThurman2012profiles_on_Miriam2014falseinteractions_query() ")
  if(tonumber(hicCellTypeToExclude)~=-1) then print("EXCLUDING "..hicCellTypeToExclude .." cell type") end
  if(tonumber(hicCellTypeToConsider)~=-1) then print("CONSIDERING only ".. hicCellTypeToConsider .." cell type") end
 
--   print("false_tuple_limit = "..comma_value(original_tuple_limit));
  
  local sql_query = dbThurman2012_column_selection_query_subpart(chrNumber);
  
  sql_query = sql_query .." FROM chromregions AS cr1   " ..
  " CROSS JOIN chromregions AS cr2   " ..
  " JOIN chromosomes AS c   " ..
  " ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr)   " ..
  " JOIN chromregionprofiles AS crp1  ON crp1.id_region = cr1.id_region   " ..
  " JOIN chromregionprofiles AS crp2  ON crp2.id_region = cr2.id_region   " ..
  " WHERE cr1.id_region <> cr2.id_region  " ..
  -- that are not in the trueinteractions   " ..
  " AND NOT EXISTS (     " ..
  "   SELECT 1     " ..
  "   FROM hic_interactions_with_labels_and_ids AS hic_inte " ..
    -- we retrieve the hic start and end of the 1st chrom region
  "   JOIN chromregions AS cr1x " ..
  "   ON hic_inte.id_region1=cr1x.id_region " ..
    -- we retrieve the hic start and end of the 2nd chrom region
  "   JOIN chromregions AS cr2x " ..
  "   ON hic_inte.id_region2=cr2x.id_region " ..
  
--     -- we retrieve the start and end of the 1st chrom region
--   "   JOIN chromregions AS cr1x " ..
--   "   ON (cr1x.chrstart = hic_cr1.chrstart " ..
--   "   AND cr1x.chrend = hic_cr1.chrend " ..
--   "   AND cr1x.id_chr = hic_cr1.id_chr) " ..
--     -- we retrieve the start and end of the 1st chrom region
--   "   JOIN chromregions AS cr2x " ..
--   "   ON (cr2x.chrstart = hic_cr2.chrstart " ..
--   "   AND cr2x.chrend = hic_cr2.chrend " ..
--   "   AND cr2x.id_chr = hic_cr2.id_chr) " ..
  
--   "   WHERE (cr1x.chrstart = cr1.chrstart " ..
--   "   AND cr1x.chrend = cr1.chrend " ..
--   "   AND cr1x.id_chr = cr1.id_chr) " ..
--   "   AND (cr2x.chrstart = cr2.chrstart " ..
--   "   AND cr2x.chrend = cr2.chrend " ..
--   "   AND cr2x.id_chr = cr2.id_chr) " ..
  
  "   WHERE (cr1x.id_region = cr1.id_region AND "..
  "   cr2x.id_region = cr2x.id_region "
  
  if (tonumber(hicCellTypeToExclude)==-1  and tonumber(hicCellTypeToConsider)==-1) then
    print("Error: select a cell type to exclude OR to consider for the Hi-C dataset. The program will stop");
    os.exit();
  end
  
  if hicCellTypeToExclude~=-1 and hicCellTypeToExclude~="-1" then
     sql_query = sql_query .. " AND celltype <> '".. hicCellTypeToExclude .."' "
  end
  if hicCellTypeToConsider~=-1 and hicCellTypeToConsider~="-1" then
     sql_query = sql_query .. " AND celltype = '".. hicCellTypeToConsider .."' "
  end  
  
  sql_query = sql_query .. " ) )   " ;
  
  if chromSel~="chr0" then sql_query = sql_query .. " AND c.name='"..chromSel.. "' "; end
  

  
  sql_query = sql_query .. " AND (cr1.chrend - cr2.chrstart <"..locus_position_limit.. ") " ..
  " AND (cr2.chrend - cr1.chrstart <"..locus_position_limit.. ") " ..
  " AND cr1.chrstart>="..tonumber(chrStart_locus)..
  " AND cr1.chrend<"..tonumber(chrEnd_locus)..
  " AND cr2.chrstart>="..tonumber(chrStart_locus)..
  " AND cr2.chrend<"..tonumber(chrEnd_locus) .." ORDER BY random() "
  -- "  ORDER BY crp1_id_region, crp2_id_region "
	
	
  -- if lengthTrues~=0 and lengthTrues~=-1 and (original_tuple_limit == -1 or original_tuple_limit == "-1") then sql_query = sql_query ..	 " LIMIT "..lengthTrues 
	-- else
    if (original_tuple_limit ~= -1 and original_tuple_limit ~= "-1") then sql_query = sql_query ..	 " LIMIT "..original_tuple_limit..";\n" 

    end

	
	-- if balancedFlag==false then sql_query = sql_query .. " LIMIT "..tonumber(lengthTrues*2) end -- JUST FOR TRYIN' @@@
	
    sql_query = sql_query .. ";"

--     print("\t\nsql_query = \n" ..sql_query);
--     io.flush();
--     os.exit();
	
    return sql_query;  
end


-- Function that creates the Sql query that reads the Thurman2012 profiles of the  Miriam2014 FALSE interactions from the PostgreSQL database, for the 4 cell types
function dbThurman2012profiles_on_Miriam2014falseinteractions_query_4_cell_types(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, original_tuple_limit, hicCellTypeToExclude1, hicCellTypeToExclude2, hicCellTypeToExclude3, hicCellTypeToExclude4, hicCellTypeToConsider1, hicCellTypeToConsider2, hicCellTypeToConsider3, hicCellTypeToConsider4)
  
  print("dbThurman2012profiles_on_Miriam2014falseinteractions_query_4_cell_types() ")
  if(tonumber(hicCellTypeToExclude1)~=-1) then print("EXCLUDING "..hicCellTypeToExclude1 .." cell type") end
  if(tonumber(hicCellTypeToExclude2)~=-1) then print("EXCLUDING "..hicCellTypeToExclude2 .." cell type") end
  if(tonumber(hicCellTypeToExclude3)~=-1) then print("EXCLUDING "..hicCellTypeToExclude3 .." cell type") end
  if(tonumber(hicCellTypeToExclude4)~=-1) then print("EXCLUDING "..hicCellTypeToExclude4 .." cell type") end
  if(tonumber(hicCellTypeToConsider1)~=-1) then print("CONSIDERING only ".. hicCellTypeToConsider1 .." cell type") end
  if(tonumber(hicCellTypeToConsider2)~=-1) then print("CONSIDERING only ".. hicCellTypeToConsider2 .." cell type") end
  if(tonumber(hicCellTypeToConsider3)~=-1) then print("CONSIDERING only ".. hicCellTypeToConsider3 .." cell type") end
  if(tonumber(hicCellTypeToConsider4)~=-1) then print("CONSIDERING only ".. hicCellTypeToConsider4 .." cell type") end
 
--   print("false_tuple_limit = "..comma_value(original_tuple_limit));
  
  local sql_query = dbThurman2012_column_selection_query_subpart(chrNumber);
  
  sql_query = sql_query .." FROM chromregions AS cr1   " ..
  " CROSS JOIN chromregions AS cr2   " ..
  " JOIN chromosomes AS c   " ..
  " ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr)   " ..
  " JOIN chromregionprofiles AS crp1  ON crp1.id_region = cr1.id_region   " ..
  " JOIN chromregionprofiles AS crp2  ON crp2.id_region = cr2.id_region   " ..
  " WHERE cr1.id_region <> cr2.id_region  " ..
  -- that are not in the trueinteractions   " ..
  " AND NOT EXISTS (     " ..
  "   SELECT 1     " ..
  "   FROM hic_interactions_with_labels_and_ids AS hic_inte " ..
    -- we retrieve the hic start and end of the 1st chrom region
  "   JOIN chromregions AS cr1x " ..
  "   ON hic_inte.id_region1=cr1x.id_region " ..
    -- we retrieve the hic start and end of the 2nd chrom region
  "   JOIN chromregions AS cr2x " ..
  "   ON hic_inte.id_region2=cr2x.id_region " ..
  
  "   WHERE (cr1x.id_region = cr1.id_region AND "..
  "   cr2x.id_region = cr2x.id_region "
  
  if (tonumber(hicCellTypeToExclude)==-1  and tonumber(hicCellTypeToConsider)==-1) then
    print("Error: select a cell type to exclude OR to consider for the Hi-C dataset. The program will stop");
    os.exit();
  end
  
  -- Cell types to exclude
  if hicCellTypeToExclude1~=-1 and hicCellTypeToExclude1~="-1" then
     sql_query = sql_query .. " AND celltype <> '".. hicCellTypeToExclude1 .."' "
  end
  if hicCellTypeToExclude2~=-1 and hicCellTypeToExclude2~="-1" then
     sql_query = sql_query .. " AND celltype <> '".. hicCellTypeToExclude2 .."' "
  end
  if hicCellTypeToExclude3~=-1 and hicCellTypeToExclude3~="-1" then
     sql_query = sql_query .. " AND celltype <> '".. hicCellTypeToExclude3 .."' "
  end
  if hicCellTypeToExclude4~=-1 and hicCellTypeToExclude4~="-1" then
     sql_query = sql_query .. " AND celltype <> '".. hicCellTypeToExclude4 .."' "
  end
  
  -- Cell types to include
  if hicCellTypeToConsider1~=-1 and hicCellTypeToConsider1~="-1" then
     sql_query = sql_query .. " AND (celltype = '".. hicCellTypeToConsider1 .."' "
  end  
  if hicCellTypeToConsider2~=-1 and hicCellTypeToConsider2~="-1" then
     sql_query = sql_query .. " OR celltype = '".. hicCellTypeToConsider2 .."' "
  end  
  if hicCellTypeToConsider3~=-1 and hicCellTypeToConsider3~="-1" then
     sql_query = sql_query .. " OR celltype = '".. hicCellTypeToConsider3 .."' "
  end  
  if hicCellTypeToConsider4~=-1 and hicCellTypeToConsider4~="-1" then
     sql_query = sql_query .. " OR celltype = '".. hicCellTypeToConsider4 .."' "
  end  
  if hicCellTypeToConsider1~=-1 and hicCellTypeToConsider1~="-1" then
     sql_query = sql_query .. " ) "
  end 
  
  sql_query = sql_query .. "  ) )  " ;
  
  if chromSel~="chr0" then sql_query = sql_query .. " AND c.name='"..chromSel.. "' "; end 

  
  sql_query = sql_query .. " AND (cr1.chrend - cr2.chrstart <"..locus_position_limit.. ") " ..
  " AND (cr2.chrend - cr1.chrstart <"..locus_position_limit.. ") " ..
  " AND cr1.chrstart>="..tonumber(chrStart_locus)..
  " AND cr1.chrend<"..tonumber(chrEnd_locus)..
  " AND cr2.chrstart>="..tonumber(chrStart_locus)..
  " AND cr2.chrend<"..tonumber(chrEnd_locus) .." ORDER BY random() "
  -- "  ORDER BY crp1_id_region, crp2_id_region "
	
    if (original_tuple_limit ~= -1 and original_tuple_limit ~= "-1") then sql_query = sql_query ..	 " LIMIT "..original_tuple_limit..";\n" 

    end

		
    sql_query = sql_query .. ";"

    -- print("\t\nsql_query = \n" ..sql_query);
    -- io.flush();
    -- os.exit();
	
    return sql_query;  
end

-- Previous function content:
-- function readDataThroughPostgreSQL_segment(chromSel, tuple_limit, locus_position_limit, balancedFlag, chrStart_locus, chrEnd_locus, execution, numberOfCellTypes, dataSource, balancedFalsePerc, uniformDistribution, dnaseExcludeColumn, hicCellTypeToExclude, hicCellTypeToConsider)

-- Function that reads a particular segment of the input dataset
-- The "dnaseExcludeColumn" parameter was initially inserted to remove one specific DNase column from the neural network, but it's not used anymore
function readDataThroughPostgreSQL_segment(chromSel, tuple_limit, locus_position_limit, balancedFlag, chrStart_locus, chrEnd_locus, execution, numberOfCellTypes, dataSource, balancedFalsePerc, uniformDistribution, dnaseExcludeColumn, hicCellTypeToExclude1, hicCellTypeToExclude2, hicCellTypeToExclude3, hicCellTypeToExclude4, hicCellTypeToConsider1, hicCellTypeToConsider2, hicCellTypeToConsider3, hicCellTypeToConsider4)

	local timeStart = os.time();
	tuple_limit = tonumber(tuple_limit);
	
	print("\n\n[^^^ readDataThroughPostgreSQL_segment ^^^]");
	print("chromSel = "..chromSel.."\n tuple_limit = "..comma_value(tuple_limit).."\n locus_position_limit = "..comma_value(locus_position_limit).."\n balancedFlag = "..tostring(balancedFlag).."\n chrStart_locus = "..comma_value(chrStart_locus).."\n chrEnd_locus = "..comma_value(chrEnd_locus));
	
	print(" execution = "..execution);
	print(" dataSource = "..dataSource);
	original_tuple_limit = tuple_limit;
	print(" original_tuple_limit = "..comma_value(original_tuple_limit));
	print("balancedFalsePerc = "..tonumber(balancedFalsePerc));
	
	print("dnaseExcludeColumn = "..dnaseExcludeColumn);
	
	io.write("\nThe Hi-C interactions of the following cell types will be excluded from the data reading:\n");
	io.flush();
	if (hicCellTypeToExclude1 ~= "-1" and hicCellTypeToExclude1 ~= -1) then io.write(hicCellTypeToExclude1.."\n"); io.flush(); end
	if (hicCellTypeToExclude2 ~= "-1" and hicCellTypeToExclude2 ~= -1) then io.write(hicCellTypeToExclude2.."\n"); io.flush(); end
	if (hicCellTypeToExclude3 ~= "-1" and hicCellTypeToExclude3 ~= -1) then io.write(hicCellTypeToExclude3.."\n"); io.flush(); end
	if (hicCellTypeToExclude4 ~= "-1" and hicCellTypeToExclude4 ~= -1) then io.write(hicCellTypeToExclude4.."\n"); io.flush(); end
	
	io.write("\nOnly the Hi-C interactions of the following cell types will be included in the data reading:\n");
	io.flush();
	if (hicCellTypeToConsider1 ~= "-1" and hicCellTypeToConsider1 ~= -1) then io.write(hicCellTypeToConsider1.."\n"); io.flush(); end
	if (hicCellTypeToConsider2 ~= "-1" and hicCellTypeToConsider2 ~= -1) then io.write(hicCellTypeToConsider2.."\n"); io.flush(); end
	if (hicCellTypeToConsider3 ~= "-1" and hicCellTypeToConsider3 ~= -1) then io.write(hicCellTypeToConsider3.."\n"); io.flush(); end
	if (hicCellTypeToConsider4 ~= "-1" and hicCellTypeToConsider4 ~= -1) then io.write(hicCellTypeToConsider4.."\n"); io.flush(); end
	io.write("\n")
	
      NEW_ARCHITECTURE=true --  TO REMOVE
      flag_array = {}
      if NEW_ARCHITECTURE==true then 
	flag_array = zero_array
	local cell_type_index1 = -1
	local cell_type_index2 = -1
	local cell_type_index3 = -1
	local cell_type_index4 = -1
	
	if (hicCellTypeToConsider1~=-1 and hicCellTypeToConsider1~="-1") then
	  cell_type_index1=retrieveNeuronIndexFromCellType(hicCellTypeToConsider1) 
	  print("hicCellTypeToConsider1 = ", hicCellTypeToConsider1)
	  flag_array[cell_type_index1] = 1.0;
	end
	if (hicCellTypeToConsider2~=-1 and hicCellTypeToConsider2~="-1") then
	  cell_type_index2=retrieveNeuronIndexFromCellType(hicCellTypeToConsider2) 
	  print("hicCellTypeToConsider2 = ", hicCellTypeToConsider2)
	  flag_array[cell_type_index2] = 1.0;
	end
	if (hicCellTypeToConsider3~=-1 and hicCellTypeToConsider3~="-1") then
	  cell_type_index3=retrieveNeuronIndexFromCellType(hicCellTypeToConsider3) 
	  print("hicCellTypeToConsider3 = ", hicCellTypeToConsider3)
	  flag_array[cell_type_index3] = 1.0;
	end
	if (hicCellTypeToConsider4~=-1 and hicCellTypeToConsider4~="-1") then
	  cell_type_index4=retrieveNeuronIndexFromCellType(hicCellTypeToConsider4) 
	  print("hicCellTypeToConsider4 = ", hicCellTypeToConsider4)
	  flag_array[cell_type_index4] = 1.0;
	end
      end
	
--       print("#flag_array = "..#flag_array)
--       print("START printVector(flag_array, flag_array)")
--       printVector(flag_array, "flag_array")
--       
--       print("END printVector(flag_array, flag_array)\n")
	
	local dnaseExcludeColumnName = "";
	if dnaseExcludeColumn>=1 and dnaseExcludeColumn<=numberOfCellTypes then
	  local columnNames = getColumnNamesOfTable("chromregionprofiles")
	  dnaseExcludeColumnName = columnNames[dnaseExcludeColumn]
	  print("EXCLUDING THE FEATURE-COLUMN "..dnaseExcludeColumnName.." number "..dnaseExcludeColumn.." among "..numberOfCellTypes);
	elseif (dnaseExcludeColumn==-1 or dnaseExcludeColumn=="-1") then
	  print("No DNase column will be excluded in the data reading");
	end
		
	local tmp = chromSel:gsub("chr","");
	local chrNumber = tmp;
	if (tmp=="X") then chrNumber = 23; 
	elseif (tmp=="Y") then chrNumber = 24;
	else chrNumber = tonumber(tmp);
	end	
	
	local sql_query_true_interactions = "";
	
	local true_tuple_limit = tonumber(tuple_limit);
	local false_tuple_limit = tonumber(tuple_limit);
	if tuple_limit~=-1 then 
	  false_tuple_limit = round(tuple_limit*tonumber(balancedFalsePerc)/100,0);
	  true_tuple_limit = tuple_limit - false_tuple_limit
	end	
	
	
	if dataSource=="Thurman2012" then 
	  sql_query_true_interactions = dbThurman2012profiles_on_Thurman2012trueinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, true_tuple_limit); 
	elseif dataSource=="Miriam2014" then 
	  sql_query_true_interactions = dbMiriam2014profiles_on_Miriam2014hicinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, true_tuple_limit); 
	elseif dataSource=="Thurman_Miriam" then
	  sql_query_true_interactions = dbThurman2012profiles_on_Miriam2014hicinteractions_query_4_cell_types(chrNumber, chromSel, chrStart_locus, chrEnd_locus, true_tuple_limit, hicCellTypeToExclude1, hicCellTypeToExclude2, hicCellTypeToExclude3, hicCellTypeToExclude4, hicCellTypeToConsider1, hicCellTypeToConsider2, hicCellTypeToConsider3, hicCellTypeToConsider4); 
	end

	-- retrieve a cursor
	local cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query_true_interactions)));	  

	-- print all rows, the rows will be indexed by field names
	local row = cur:fetch ({}, "a");
	-- print(string.format("first_chrname-first_chrstart-first_chrend\tsecond_chrname-second_chrstart-second_chrend\n"))
	
	if dnaseExcludeColumn>=1 and dnaseExcludeColumn <=numberOfCellTypes then
	  numberOfCellTypes = numberOfCellTypes -1
	end
	print("database_management.lua: numberOfCellTypes = "..numberOfCellTypes)
	
        first_profile_initial = 4
	first_profile_finish = numberOfCellTypes+3 
	second_profile_initial = numberOfCellTypes+4
	second_profile_finish = (numberOfCellTypes*2)+3
	last_index = 4+(numberOfCellTypes*2);

	local dnaseDataTableTrue = {}
	i = 1;

	print("Readin' the true interaction rows from the database");
	while row do	
	  
	  if i%1000==0 then io.write("(i="..comma_value(i)..") "); io.flush(); end
	  
	  local targetValue = 1
	  dnaseDataTableTrue[i] = sqlRowRetrieverCommand(row, dataSource, targetValue, first_profile_initial, second_profile_finish, dnaseExcludeColumn, dnaseExcludeColumnName, numberOfCellTypes)
	  
	  -- WE DO NOT SELECT "row.source"
	  row = cur:fetch(row, "a");
	  i = i + 1;
	end	
	--print("#dnaseDataTableTrue=".. comma_value(#dnaseDataTableTrue));
	
	  -- ORIGINAL CHECK: TO REESTABLISH
	  -- if( #dnaseDataTableTrue<100 and tuple_limit==-1 and execution~="JUST-TESTING") then print("Not enough tuples ("..(#dnaseDataTableTrue).."<100) read in the database; the program is going to stop."); os.exit(); end
	
	  if( #dnaseDataTableTrue==0 and execution~="JUST-TESTING" and execution~="OPTIMIZATION-HELD-OUT" and original_tuple_limit==-1) then print("Not enough tuples ("..(#dnaseDataTableTrue).."==0) read in the database; the program is going to stop. RE-ESTABLISH THE 100 CONTROL!!!"); os.exit(); end
	
	 local length = -1
	 local lengthTrues = -1
	  
	if #dnaseDataTableTrue>0 then	
	  length = #dnaseDataTableTrue;
	  lengthTrues = #dnaseDataTableTrue;
	  width = (#dnaseDataTableTrue[1])[1];	
	else	
	  length = 0
	  lengthTrues = 0	
	end
	
	 print('#dnaseDataTableTrue = '..comma_value(#dnaseDataTableTrue));
	 io.write(' tuple_limit = '..comma_value(tuple_limit));
	 io.write(' true_tuple_limit = '..comma_value(true_tuple_limit));
	 io.write(' false_tuple_limit = '..comma_value(false_tuple_limit).."\n");
	 io.flush();
	 
	if original_tuple_limit~=-1 and #dnaseDataTableTrue < true_tuple_limit then
	  print("PROBLEM: not enough true interactions read (just "..#dnaseDataTableTrue..")");	  
	  
	  -- false_tuple_limit = #dnaseDataTableTrue
	  
	  false_tuple_limit = round((#dnaseDataTableTrue/((100-balancedFalsePerc)/100))*balancedFalsePerc/100,0);
	  print("The program will set false_tuple_limit = "..false_tuple_limit);
	  
	  -- os.exit()
	end

	--printTime(timeStart, "first Sql execution duration");
	timeSecondSqlExecution = os.time()
	
	
	-- -- FALSE INTERACTIONS in the dnaseDataTableFalse 
	local sql_query_false_interactions = "";
	
	if dataSource=="Thurman2012" then 
	  sql_query_false_interactions = dbThurman2012profiles_on_Thurman2012falseinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, false_tuple_limit);
	elseif dataSource=="Miriam2014" then 
	  sql_query_false_interactions = dbMiriam2014profiles_on_Miriam2014falseinteractions_query(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, false_tuple_limit);
	elseif dataSource=="Thurman_Miriam" then
	  sql_query_false_interactions = dbThurman2012profiles_on_Miriam2014falseinteractions_query_4_cell_types(chrNumber, chromSel, chrStart_locus, chrEnd_locus, locus_position_limit, lengthTrues, balancedFlag, false_tuple_limit, hicCellTypeToExclude1, hicCellTypeToExclude2, hicCellTypeToExclude3, hicCellTypeToExclude4, hicCellTypeToConsider1, hicCellTypeToConsider2, hicCellTypeToConsider3, hicCellTypeToConsider4);
	end
		  
	-- retrieve a cursor
	cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query_false_interactions)));
		 
	-- print all rows, the rows will be indexed by field names
	row = cur:fetch ({}, "a");
	-- print(string.format("first_chrname-first_chrstart-first_chrend\tsecond_chrname-second_chrstart-second_chrend\n"))
	 
		 
	local dnaseDataTableFalse = {}
	i = 1;
	
	print("Readin' the false interaction rows from the database");
	while row do	   

	  if i%1000==0 then io.write("(i="..comma_value(i)..") "); io.flush(); end
	   
	   local targetValue = 0
	   dnaseDataTableFalse[i] = sqlRowRetrieverCommand(row, dataSource, targetValue, first_profile_initial, second_profile_finish, dnaseExcludeColumn, dnaseExcludeColumnName, numberOfCellTypes)	   
	    -- WE DO NOT SELECT "row.source"
	 	   
	   row = cur:fetch(row, "a");
	   i = i + 1;	   
	end
	 
	 
	-- close everything
	cur:close(); 
	-- closeGlobalDbConnection()	 
	 
	print('length = #dnaseDataTableFalse '..comma_value(#dnaseDataTableFalse));
	printTime(timeStart, "PostgreSQL data reading")
	
	io.write("\n");
	
	local zeroCount = 1
	local oneCount = 1
	local dnaseDataTable = {}
	local newDatasetTable = {}
	

	local dataset_firstChromRegion = {};
	local dataset_secondChromRegion = {};
	local targetVector = {};

	local completeTable = {}
	local dataset = {}	
	generalMaxValue = -1
	
	local dnaseDataTable_only_IDs = {}
	
	local dataset_firstChromRegion_newArchi = {};
	local dataset_secondChromRegion_newArchi = {};

	-- IF AT LEAST ONE "ONE" IS PRESENT
	
	local dnaseDataTable_length = -1;
	
	if #dnaseDataTableTrue > 0 then
	
	  true_dnaseDataTable_length = #dnaseDataTableTrue;
	  print('true_dnaseDataTable_length = #dnaseDataTableTrue '..comma_value(true_dnaseDataTable_length));
	  false_dnaseDataTable_length = #dnaseDataTableFalse;
	  print('false_dnaseDataTable_length = #dnaseDataTableFalse '..comma_value(false_dnaseDataTable_length));
	  dnaseDataTable_length = false_dnaseDataTable_length+true_dnaseDataTable_length;
	  print('dnaseDataTable_length = #dnaseDataTable '..comma_value(dnaseDataTable_length));
	  sparsity = round(true_dnaseDataTable_length*100/dnaseDataTable_length,2);
	  print("sparsity of the input vector: "..sparsity.."%");
	  
	  zeroSpanSize = math.floor(dnaseDataTable_length/true_dnaseDataTable_length);
	  print("zeroSpanSize = "..zeroSpanSize);
	  
	  local printCount = 0;
	 	  
	  i = 1;
	  print('input matrix creation:');
	  io.flush();
	  
	  for i=1,dnaseDataTable_length do
	    
	    local rate = round(i*100/dnaseDataTable_length,2);
	    if((rate*10)%10==0) then io.write(rate.."% "); io.flush(); end
	    if(rate%10==0) then io.write("\n"); io.flush(); end
	    printCount = printCount + 1;
	  
	  
	    dnaseDataTable[i] = {};
	    dnaseDataTable_only_IDs[i] = {};
	    
	    if uniformDistribution == true then
		-- LINEAR DISTRIBUTION OF ZEROS AND ONES: ONE 1 EVERY zeroSpanSize
		if i == 1 then print("(Uniform distribution of ones among the dataset)"); end
		if i%zeroSpanSize==0 and oneCount<=#dnaseDataTableTrue then
		  dnaseDataTable[i] = dnaseDataTableTrue[oneCount];
		  oneCount = oneCount + 1;
		elseif zeroCount<=#dnaseDataTableFalse then
		  dnaseDataTable[i] = dnaseDataTableFalse[zeroCount];
		  zeroCount = zeroCount + 1;
		else 
		  break
		end		
	  else		
		-- 	FIRST ALL THE ONES, THEN ALL THE ZEROS
		if i == 1 then print("(Distribution: first all the ones, and then all the zeros)"); end
		if oneCount<=#dnaseDataTableTrue then
		  dnaseDataTable[i] = dnaseDataTableTrue[oneCount];
		  oneCount = oneCount + 1;
		else -- if oneCount>#dnaseDataTableTrue then
		  dnaseDataTable[i] = dnaseDataTableFalse[zeroCount];
		  zeroCount = zeroCount + 1;
		--  else 
		--  break
		end	
	    end
	    
	    -- only chromosome number, first chrom region ID, and second chrom region ID
	    dnaseDataTable_only_IDs[i] = dnaseDataTable[i][{{1,3}}];
	    
	    
	      local tempTens = torch.Tensor(dnaseDataTable[i]);
    
	      dataset_firstChromRegion[i] = tempTens[{{first_profile_initial,first_profile_finish}}];
	      dataset_secondChromRegion[i] = tempTens[{{second_profile_initial,second_profile_finish}}];
	      targetVector[i] = torch.Tensor{dnaseDataTable[i][last_index]};      
	      
	      if torch.max(dataset_firstChromRegion[i]) > generalMaxValue then 
		generalMaxValue = torch.max(dataset_firstChromRegion[i])
	      end
	      if torch.max(dataset_secondChromRegion[i]) > generalMaxValue then 
		generalMaxValue = torch.max(dataset_secondChromRegion[i])
	      end
	      
	      local firstMean = round(torch.mean(dataset_firstChromRegion[i]),2);
	      local secondMean = round(torch.mean(dataset_secondChromRegion[i]),2);
	      local meanOfMeans = round(((firstMean + secondMean)/2),2)
	      
	      if PRINT_AVERAGES then  
		io.write("["..i.."] firstChromReg mean = "..firstMean);
		io.write(" secondChromReg mean = "..secondMean);
		io.write("\t meanOfMeans = "..meanOfMeans);
		io.write("\n");
		io.flush();
	      end
	      

	      completeTable[i] = {};
	      completeTable[i] = torch.Tensor(dnaseDataTable[i])[{{first_profile_initial,last_index-1}}];
	      -- previously there was 3 instead of first_profile_initial
	      
	      dataset[i] = {completeTable[i], targetVector[i]};	     
	      io.flush();
	      collectgarbage();
  
	      -- print(#(dataset_firstChromRegion[i]))
	      -- print(#(dataset_secondChromRegion[i]))
	      
	      -- aaa = dataset_firstChromRegion[i]
	      
	      if NEW_ARCHITECTURE == true then
		dataset_firstChromRegion_newArchi[i] = tensorArrayConcatByPos(dataset_firstChromRegion[i], flag_array)
		dataset_secondChromRegion_newArchi[i] = tensorArrayConcatByPos(dataset_secondChromRegion[i], flag_array)
	      end
	    
	  end
	  
	else  -- if there are no true interactions
	  
	  false_dnaseDataTable_length = #dnaseDataTableFalse;
	  dnaseDataTable_length = false_dnaseDataTable_length
	  print('dnaseDataTable_length = #dnaseDataTable '..comma_value(dnaseDataTable_length));
	  
	  -- dnaseDataTable is the starting point	   
	  
	  print('input matrix creation:');
	  io.flush();
	  for i=1,dnaseDataTable_length do
	    local rate = round(i*100/dnaseDataTable_length,2);
	    if(dnaseDataTable_length>= 300  and i%300==0) then io.write(rate.."% "); io.flush(); end
	    if(dnaseDataTable_length< 300  and i%10==0) then io.write(rate.."% "); io.flush(); end
	  
	    dnaseDataTable[i] = dnaseDataTableFalse[i]    
	    
    	    dnaseDataTable_only_IDs[i] = (dnaseDataTable[i])[{{1,3}}];
            tempTens = torch.Tensor(dnaseDataTable[i])
    
            dataset_firstChromRegion[i] = tempTens[{{first_profile_initial,first_profile_finish}}]
	    dataset_secondChromRegion[i] = tempTens[{{second_profile_initial,second_profile_finish}}]
	    targetVector[i] = torch.Tensor{dnaseDataTable[i][last_index]}

	    if torch.max(dataset_firstChromRegion[i]) > generalMaxValue then 
		generalMaxValue = torch.max(dataset_firstChromRegion[i])
	    end
	    if torch.max(dataset_secondChromRegion[i]) > generalMaxValue then 
		generalMaxValue = torch.max(dataset_secondChromRegion[i])
	    end
	    
	    if NEW_ARCHITECTURE == true then
		dataset_firstChromRegion_newArchi[i] = tensorArrayConcatByPos(dataset_firstChromRegion[i], flag_array)
		dataset_secondChromRegion_newArchi[i] = tensorArrayConcatByPos(dataset_secondChromRegion[i], flag_array)
	    end
	    
	     if PRINT_AVERAGES then  
	      io.write("["..i.."] firstChromReg aver. value = "..round(torch.mean(dataset_firstChromRegion[i]),2));
	      print(" secondChromReg aver. value = "..round(torch.mean(dataset_secondChromRegion[i]),2));
	      io.flush();
	     end
	    
	    completeTable[i] = {}
	    completeTable[i] = torch.Tensor(dnaseDataTable[i])[{{first_profile_initial,last_index-1}}]
	    -- previously there was 3 instead of first_profile_initial

	    dataset[i] = {completeTable[i], targetVector[i]};
	    collectgarbage();	    
	  end	
	end
	
	print("\ngeneralMaxValue = "..comma_value(generalMaxValue));
	
	if DATA_NORMALIZATION == true then
	  for i=1,dnaseDataTable_length do
	    
	    dataset_firstChromRegion[i] = dataset_firstChromRegion[i] / generalMaxValue
	    dataset_secondChromRegion[i] = dataset_secondChromRegion[i] / generalMaxValue
	    
	  end
	end
	
      print("\n#dnaseDataTable = "..comma_value(#dnaseDataTable));
      print("\n#dnaseDataTable_length = ".. comma_value(dnaseDataTable_length));      
      if #dnaseDataTable~=dnaseDataTable_length then print("Dimension error because of #dnaseDataTable~=dnaseDataTable_length.\n The program will stop."); os.exit(); end

      if #dnaseDataTable == 0 then 
	print("Attention: there are no true interactions neither false interactions in the area between chrStart_locus="..comma_value(chrStart_locus).." and chrEnd_locus="..comma_value(chrEnd_locus).." of "..chromSel);
	print("The program will stop");
	os.exit();
      end

      printTime(timeStart, "PostgreSQL data reading and matrix creation")
      print("[vvv readDataThroughPostgreSQL_segment vvv]\n\n\n");
      
      -- return {lengthTrues, dnaseDataTable};	
      
      if NEW_ARCHITECTURE == false then
      	return {lengthTrues, dnaseDataTable, dataset_firstChromRegion, dataset_secondChromRegion, targetVector, completeTable, dataset, dnaseDataTable_only_IDs};
      else
	return {lengthTrues, dnaseDataTable, dataset_firstChromRegion, dataset_secondChromRegion, targetVector, completeTable, dataset, dnaseDataTable_only_IDs, dataset_firstChromRegion_newArchi, dataset_secondChromRegion_newArchi};
      end
end

-- function that reads a matrix of chrom region numbers and ID's, and saves them into an SQL table
function saveChromRegionPairsToDB(dnaseDataTable_only_IDs);
  
end


-- Function that reads all the input dataset
function readDataThroughPostgreSQL(chromSel, tuple_limit, locus_position_limit, balancedFlag)

  
	timeStart = os.time();
	
	print("readThurman2012dataThroughPostgreSQL(chromSel="..chromSel..", tuple_limit="..tuple_limit..", locus_position_limit="..locus_position_limit..", balancedFlag="..tostring(balancedFlag)..")")
	original_tuple_limit = tuple_limit
  
	flagNoLimits = false;
	if tuple_limit == -1 then flagNoLimits = true; end
	
	tmp = chromSel:gsub("chr","");
	chrNumber = tmp;
	if (tmp=="X") then chrNumber = 23; 
	elseif (tmp=="Y") then chrNumber = 24;
	else chrNumber = tonumber(tmp);
	end
	

	-- TRUE INTERACTIONS in the dnaseDataTableTrue 

	sql_query = "\nSELECT ".. chrNumber  .." AS name, cr1.id_region AS cr1_id_region, cr2.id_region AS cr2_id_region, cr1.a549_ds14289 AS cr1_a549_ds14289,  cr1.ag10803_ds12374 AS cr1_ag10803_ds12374,  cr1.aoaf_ds13513 AS cr1_aoaf_ds13513,  cr1.cd14_ds17215 AS cr1_cd14_ds17215,  cr1.cd19_ds17186 AS cr1_cd19_ds17186,  cr1.cd20_ds18208 AS cr1_cd20_ds18208,  cr1.cd34_ds12274 AS cr1_cd34_ds12274,  cr1.cd3_cordblood_ds17706 AS cr1_cd3_cordblood_ds17706,  cr1.cd3_ds17198 AS cr1_cd3_ds17198,  cr1.cd4_ds17212 AS cr1_cd4_ds17212,  cr1.cd4pos_n_ds14108 AS cr1_cd4pos_n_ds14108,  cr1.cd56_ds17189 AS cr1_cd56_ds17189,  cr1.cd8_ds17203 AS cr1_cd8_ds17203,  cr1.fbrain_ds11872 AS cr1_fbrain_ds11872,  cr1.fheart_ds12531 AS cr1_fheart_ds12531,  cr1.fintestine_lg_ds17313 AS cr1_fintestine_lg_ds17313,  cr1.fkidney_ds20786 AS cr1_fkidney_ds20786,  cr1.flung_ds14724 AS cr1_flung_ds14724,  cr1.fmuscle_leg_ds20239 AS cr1_fmuscle_leg_ds20239,  cr1.fplacenta_ds20346 AS cr1_fplacenta_ds20346,  cr1.fskin_fibro_leg_r_quad_ds19943 AS cr1_fskin_fibro_leg_r_quad_ds19943,  cr1.fspinal_cord_ds20351 AS cr1_fspinal_cord_ds20351,  cr1.fstomach_ds17878 AS cr1_fstomach_ds17878,  cr1.fthymus_ds20341 AS cr1_fthymus_ds20341,  cr1.gm06990_ds7748 AS cr1_gm06990_ds7748,  cr1.gm12865_ds12436 AS cr1_gm12865_ds12436,  cr1.haepic_ds12663 AS cr1_haepic_ds12663,  cr1.hah_ds15192 AS cr1_hah_ds15192,  cr1.hasp_ds14790 AS cr1_hasp_ds14790,  cr1.hcfaa_ds13480 AS cr1_hcfaa_ds13480,  cr1.hcf_ds12501 AS cr1_hcf_ds12501,  cr1.hcm_ds12599 AS cr1_hcm_ds12599,  cr1.hcpepic_ds12447 AS cr1_hcpepic_ds12447,  cr1.heepic_ds12763 AS cr1_heepic_ds12763,  cr1.hepg2_ds7764 AS cr1_hepg2_ds7764,  cr1.hesct0_ds11909 AS cr1_hesct0_ds11909,  cr1.hff_ds15115 AS cr1_hff_ds15115,  cr1.hgf_ds11752 AS cr1_hgf_ds11752,  cr1.hipepic_ds12684 AS cr1_hipepic_ds12684,  cr1.hmf_ds13368 AS cr1_hmf_ds13368,  cr1.hmvec_dblad_ds13337 AS cr1_hmvec_dblad_ds13337,  cr1.hmvec_dblneo_ds13242 AS cr1_hmvec_dblneo_ds13242,  cr1.hmvec_dlyneo_ds13150 AS cr1_hmvec_dlyneo_ds13150,  cr1.hmvec_lbl_ds13372 AS cr1_hmvec_lbl_ds13372,  cr1.hmvec_lly_ds13185 AS cr1_hmvec_lly_ds13185,  cr1.hpaf_ds13411 AS cr1_hpaf_ds13411,  cr1.hpdlf_ds13573 AS cr1_hpdlf_ds13573,  cr1.hpf_ds13390 AS cr1_hpf_ds13390,  cr1.hrce_ds10666 AS cr1_hrce_ds10666,  cr1.hsmm_ds14426 AS cr1_hsmm_ds14426,  cr1.hth17_ds11039 AS cr1_hth17_ds11039,  cr1.hth1_ds7840 AS cr1_hth1_ds7840,  cr1.hth2ds17597 AS cr1_hth2ds17597,  cr1.htr_ds14702 AS cr1_htr_ds14702,  cr1.huvec_ds10060 AS cr1_huvec_ds10060,  cr1.hvmf_ds13981 AS cr1_hvmf_ds13981,  cr1.imr90_ds13219 AS cr1_imr90_ds13219,  cr1.ips_19_11_ds15153 AS cr1_ips_19_11_ds15153,  cr1.ith1_ds18018 AS cr1_ith1_ds18018,  cr1.ith2_ds17603 AS cr1_ith2_ds17603,  cr1.k562_ds9767 AS cr1_k562_ds9767,  cr1.lhcn_m2_ds20548 AS cr1_lhcn_m2_ds20548,  cr1.m059j_ds20493 AS cr1_m059j_ds20493,  cr1.mesendoderm_ds19310 AS cr1_mesendoderm_ds19310,  cr1.msc_ds21042 AS cr1_msc_ds21042,  cr1.nb4_ds12543 AS cr1_nb4_ds12543,  cr1.nha_ds12800 AS cr1_nha_ds12800,  cr1.nhdf_ad_ds12863 AS cr1_nhdf_ad_ds12863,  cr1.nhdf_neo_ds11923 AS cr1_nhdf_neo_ds11923,  cr1.nhlf_ds12829 AS cr1_nhlf_ds12829,  cr1.psoas_muscle_ds20325 AS cr1_psoas_muscle_ds20325,  cr1.rpmi_7951_ds20909 AS cr1_rpmi_7951_ds20909,  cr1.saec_ds10518 AS cr1_saec_ds10518,  cr1.skin_fibroblasts_ds18224 AS cr1_skin_fibroblasts_ds18224,  cr1.skin_keratinocytes_ds18692 AS cr1_skin_keratinocytes_ds18692,  cr1.skin_melanocytes_ds18590 AS cr1_skin_melanocytes_ds18590,  cr1.skmc_ds11949 AS cr1_skmc_ds11949,  cr1.sknsh_ds8482 AS cr1_sknsh_ds8482,  cr1.small_bowel_mucosa_ds20770 AS cr1_small_bowel_mucosa_ds20770,  cr1.t_47d_ds19794 AS cr1_t_47d_ds19794,  cr1.trophoblast_ds19317 AS cr1_trophoblast_ds19317,  cr1.vhmec_ds18406 AS cr1_vhmec_ds18406,  cr2.a549_ds14289 AS cr2_a549_ds14289,  cr2.ag10803_ds12374 AS cr2_ag10803_ds12374,  cr2.aoaf_ds13513 AS cr2_aoaf_ds13513,  cr2.cd14_ds17215 AS cr2_cd14_ds17215,  cr2.cd19_ds17186 AS cr2_cd19_ds17186,  cr2.cd20_ds18208 AS cr2_cd20_ds18208,  cr2.cd34_ds12274 AS cr2_cd34_ds12274,  cr2.cd3_cordblood_ds17706 AS cr2_cd3_cordblood_ds17706,  cr2.cd3_ds17198 AS ";
	
	sql_query = sql_query .." cr2_cd3_ds17198,  cr2.cd4_ds17212 AS cr2_cd4_ds17212,  cr2.cd4pos_n_ds14108 AS cr2_cd4pos_n_ds14108,  cr2.cd56_ds17189 AS cr2_cd56_ds17189,  cr2.cd8_ds17203 AS cr2_cd8_ds17203,  cr2.fbrain_ds11872 AS cr2_fbrain_ds11872,  cr2.fheart_ds12531 AS cr2_fheart_ds12531,  cr2.fintestine_lg_ds17313 AS cr2_fintestine_lg_ds17313,  cr2.fkidney_ds20786 AS cr2_fkidney_ds20786,  cr2.flung_ds14724 AS cr2_flung_ds14724,  cr2.fmuscle_leg_ds20239 AS cr2_fmuscle_leg_ds20239,  cr2.fplacenta_ds20346 AS cr2_fplacenta_ds20346,  cr2.fskin_fibro_leg_r_quad_ds19943 AS cr2_fskin_fibro_leg_r_quad_ds19943,  cr2.fspinal_cord_ds20351 AS cr2_fspinal_cord_ds20351,  cr2.fstomach_ds17878 AS cr2_fstomach_ds17878,  cr2.fthymus_ds20341 AS cr2_fthymus_ds20341,  cr2.gm06990_ds7748 AS cr2_gm06990_ds7748,  cr2.gm12865_ds12436 AS cr2_gm12865_ds12436,  cr2.haepic_ds12663 AS cr2_haepic_ds12663,  cr2.hah_ds15192 AS cr2_hah_ds15192,  cr2.hasp_ds14790 AS cr2_hasp_ds14790,  cr2.hcfaa_ds13480 AS cr2_hcfaa_ds13480,  cr2.hcf_ds12501 AS cr2_hcf_ds12501,  cr2.hcm_ds12599 AS cr2_hcm_ds12599,  cr2.hcpepic_ds12447 AS cr2_hcpepic_ds12447,  cr2.heepic_ds12763 AS cr2_heepic_ds12763,  cr2.hepg2_ds7764 AS cr2_hepg2_ds7764,  cr2.hesct0_ds11909 AS cr2_hesct0_ds11909,  cr2.hff_ds15115 AS cr2_hff_ds15115,  cr2.hgf_ds11752 AS cr2_hgf_ds11752,  cr2.hipepic_ds12684 AS cr2_hipepic_ds12684,  cr2.hmf_ds13368 AS cr2_hmf_ds13368,  cr2.hmvec_dblad_ds13337 AS cr2_hmvec_dblad_ds13337,  cr2.hmvec_dblneo_ds13242 AS cr2_hmvec_dblneo_ds13242,  cr2.hmvec_dlyneo_ds13150 AS cr2_hmvec_dlyneo_ds13150,  cr2.hmvec_lbl_ds13372 AS cr2_hmvec_lbl_ds13372,  cr2.hmvec_lly_ds13185 AS cr2_hmvec_lly_ds13185,  cr2.hpaf_ds13411 AS cr2_hpaf_ds13411,  cr2.hpdlf_ds13573 AS cr2_hpdlf_ds13573,  cr2.hpf_ds13390 AS cr2_hpf_ds13390,  cr2.hrce_ds10666 AS cr2_hrce_ds10666,  cr2.hsmm_ds14426 AS cr2_hsmm_ds14426,  cr2.hth17_ds11039 AS cr2_hth17_ds11039,  cr2.hth1_ds7840 AS cr2_hth1_ds7840,  cr2.hth2ds17597 AS cr2_hth2ds17597,  cr2.htr_ds14702 AS cr2_htr_ds14702,  cr2.huvec_ds10060 AS cr2_huvec_ds10060,  cr2.hvmf_ds13981 AS cr2_hvmf_ds13981,  cr2.imr90_ds13219 AS cr2_imr90_ds13219,  cr2.ips_19_11_ds15153 AS cr2_ips_19_11_ds15153,  cr2.ith1_ds18018 AS cr2_ith1_ds18018,  cr2.ith2_ds17603 AS cr2_ith2_ds17603,  cr2.k562_ds9767 AS cr2_k562_ds9767,  cr2.lhcn_m2_ds20548 AS cr2_lhcn_m2_ds20548,  cr2.m059j_ds20493 AS cr2_m059j_ds20493,  cr2.mesendoderm_ds19310 AS cr2_mesendoderm_ds19310,  cr2.msc_ds21042 AS cr2_msc_ds21042,  cr2.nb4_ds12543 AS cr2_nb4_ds12543,  cr2.nha_ds12800 AS cr2_nha_ds12800,  cr2.nhdf_ad_ds12863 AS cr2_nhdf_ad_ds12863,  cr2.nhdf_neo_ds11923 AS cr2_nhdf_neo_ds11923,  cr2.nhlf_ds12829 AS cr2_nhlf_ds12829,  cr2.psoas_muscle_ds20325 AS cr2_psoas_muscle_ds20325,  cr2.rpmi_7951_ds20909 AS cr2_rpmi_7951_ds20909,  cr2.saec_ds10518 AS cr2_saec_ds10518,  cr2.skin_fibroblasts_ds18224 AS cr2_skin_fibroblasts_ds18224,  cr2.skin_keratinocytes_ds18692 AS cr2_skin_keratinocytes_ds18692,  cr2.skin_melanocytes_ds18590 AS cr2_skin_melanocytes_ds18590,  cr2.skmc_ds11949 AS cr2_skmc_ds11949,  cr2.sknsh_ds8482 AS cr2_sknsh_ds8482,  cr2.small_bowel_mucosa_ds20770 AS cr2_small_bowel_mucosa_ds20770,  cr2.t_47d_ds19794 AS cr2_t_47d_ds19794,  cr2.trophoblast_ds19317 AS cr2_trophoblast_ds19317,  cr2.vhmec_ds18406  AS cr2_vhmec_ds18406 " ..
	" FROM trueinteractions AS ti " ..
	" JOIN chromregionprofiles AS cr1 " ..
	" ON ti.id_region1=cr1.id_region " ..
	" JOIN chromregionprofiles AS cr2 " ..
	" ON ti.id_region2=cr2.id_region " ..
	" JOIN chromregions AS cr " ..
	" ON ti.id_region1=cr.id_region " ..
	" JOIN chromosomes AS c " ..
	" ON c.id_chr=cr.id_chr " ..
	" WHERE c.name='"..chromSel.. "'".." ORDER BY random() "
	--	" AND (cr1.chrend - cr2.chrstart <"..locus_position_limit.. ") " ..
--	" AND (cr2.chrend - cr1.chrstart <"..locus_position_limit.. ") "
	-- " ORDER BY cr1_id_region, cr2_id_region "
	
	if flagNoLimits==false then sql_query = sql_query .. " LIMIT "..tuple_limit;

	sql_query = sql_query..";\n"
 end


	-- print("\tsql_query: \n"..sql_query); io.flush();

	-- retrieve a cursor
	cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query)));
	  
	-- print all rows, the rows will be indexed by field names
	row = cur:fetch ({}, "a");
	-- print(string.format("first_chrname-first_chrstart-first_chrend\tsecond_chrname-second_chrstart-second_chrend\n"))


	dnaseDataTableTrue = {}
	i = 1;

	while row do
	  
	  --dnaseDataTableTrue[i] = {};
	  
	  dnaseDataTableTrue[i] = torch.Tensor({tonumber(row.name), tonumber(row.cr1_id_region), tonumber(row.cr2_id_region), tonumber(row.cr1_a549_ds14289), tonumber(row.cr1_ag10803_ds12374), tonumber(row.cr1_aoaf_ds13513), tonumber(row.cr1_cd14_ds17215), tonumber(row.cr1_cd19_ds17186), tonumber(row.cr1_cd20_ds18208), tonumber(row.cr1_cd34_ds12274), tonumber(row.cr1_cd3_cordblood_ds17706), tonumber(row.cr1_cd3_ds17198), tonumber(row.cr1_cd4_ds17212), tonumber(row.cr1_cd4pos_n_ds14108), tonumber(row.cr1_cd56_ds17189), tonumber(row.cr1_cd8_ds17203), tonumber(row.cr1_fbrain_ds11872), tonumber(row.cr1_fheart_ds12531), tonumber(row.cr1_fintestine_lg_ds17313), tonumber(row.cr1_fkidney_ds20786), tonumber(row.cr1_flung_ds14724), tonumber(row.cr1_fmuscle_leg_ds20239), tonumber(row.cr1_fplacenta_ds20346), tonumber(row.cr1_fskin_fibro_leg_r_quad_ds19943), tonumber(row.cr1_fspinal_cord_ds20351), tonumber(row.cr1_fstomach_ds17878), tonumber(row.cr1_fthymus_ds20341), tonumber(row.cr1_gm06990_ds7748), tonumber(row.cr1_gm12865_ds12436), tonumber(row.cr1_haepic_ds12663), tonumber(row.cr1_hah_ds15192), tonumber(row.cr1_hasp_ds14790), tonumber(row.cr1_hcfaa_ds13480), tonumber(row.cr1_hcf_ds12501), tonumber(row.cr1_hcm_ds12599), tonumber(row.cr1_hcpepic_ds12447), tonumber(row.cr1_heepic_ds12763), tonumber(row.cr1_hepg2_ds7764), tonumber(row.cr1_hesct0_ds11909), tonumber(row.cr1_hff_ds15115), tonumber(row.cr1_hgf_ds11752), tonumber(row.cr1_hipepic_ds12684), tonumber(row.cr1_hmf_ds13368), tonumber(row.cr1_hmvec_dblad_ds13337), tonumber(row.cr1_hmvec_dblneo_ds13242), tonumber(row.cr1_hmvec_dlyneo_ds13150), tonumber(row.cr1_hmvec_lbl_ds13372), tonumber(row.cr1_hmvec_lly_ds13185), tonumber(row.cr1_hpaf_ds13411), tonumber(row.cr1_hpdlf_ds13573), tonumber(row.cr1_hpf_ds13390), tonumber(row.cr1_hrce_ds10666), tonumber(row.cr1_hsmm_ds14426), tonumber(row.cr1_hth17_ds11039), tonumber(row.cr1_hth1_ds7840), tonumber(row.cr1_hth2ds17597), tonumber(row.cr1_htr_ds14702), tonumber(row.cr1_huvec_ds10060), tonumber(row.cr1_hvmf_ds13981), tonumber(row.cr1_imr90_ds13219), tonumber(row.cr1_ips_19_11_ds15153), tonumber(row.cr1_ith1_ds18018), tonumber(row.cr1_ith2_ds17603), tonumber(row.cr1_k562_ds9767), tonumber(row.cr1_lhcn_m2_ds20548), tonumber(row.cr1_m059j_ds20493), tonumber(row.cr1_mesendoderm_ds19310), tonumber(row.cr1_msc_ds21042), tonumber(row.cr1_nb4_ds12543), tonumber(row.cr1_nha_ds12800), tonumber(row.cr1_nhdf_ad_ds12863), tonumber(row.cr1_nhdf_neo_ds11923), tonumber(row.cr1_nhlf_ds12829), tonumber(row.cr1_psoas_muscle_ds20325), tonumber(row.cr1_rpmi_7951_ds20909), tonumber(row.cr1_saec_ds10518), tonumber(row.cr1_skin_fibroblasts_ds18224), tonumber(row.cr1_skin_keratinocytes_ds18692), tonumber(row.cr1_skin_melanocytes_ds18590), tonumber(row.cr1_skmc_ds11949), tonumber(row.cr1_sknsh_ds8482), tonumber(row.cr1_small_bowel_mucosa_ds20770), tonumber(row.cr1_t_47d_ds19794), tonumber(row.cr1_trophoblast_ds19317), tonumber(row.cr1_vhmec_ds18406), 
	    tonumber(row.cr2_a549_ds14289), tonumber(row.cr2_ag10803_ds12374), tonumber(row.cr2_aoaf_ds13513), tonumber(row.cr2_cd14_ds17215), tonumber(row.cr2_cd19_ds17186), tonumber(row.cr2_cd20_ds18208), tonumber(row.cr2_cd34_ds12274), tonumber(row.cr2_cd3_cordblood_ds17706), tonumber(row.cr2_cd3_ds17198), tonumber(row.cr2_cd4_ds17212), tonumber(row.cr2_cd4pos_n_ds14108), tonumber(row.cr2_cd56_ds17189), tonumber(row.cr2_cd8_ds17203), tonumber(row.cr2_fbrain_ds11872), tonumber(row.cr2_fheart_ds12531), tonumber(row.cr2_fintestine_lg_ds17313), tonumber(row.cr2_fkidney_ds20786), tonumber(row.cr2_flung_ds14724), tonumber(row.cr2_fmuscle_leg_ds20239), tonumber(row.cr2_fplacenta_ds20346), tonumber(row.cr2_fskin_fibro_leg_r_quad_ds19943), tonumber(row.cr2_fspinal_cord_ds20351), tonumber(row.cr2_fstomach_ds17878), tonumber(row.cr2_fthymus_ds20341), tonumber(row.cr2_gm06990_ds7748), tonumber(row.cr2_gm12865_ds12436), tonumber(row.cr2_haepic_ds12663), tonumber(row.cr2_hah_ds15192), tonumber(row.cr2_hasp_ds14790), tonumber(row.cr2_hcfaa_ds13480), tonumber(row.cr2_hcf_ds12501), tonumber(row.cr2_hcm_ds12599), tonumber(row.cr2_hcpepic_ds12447), tonumber(row.cr2_heepic_ds12763), tonumber(row.cr2_hepg2_ds7764), tonumber(row.cr2_hesct0_ds11909), tonumber(row.cr2_hff_ds15115), tonumber(row.cr2_hgf_ds11752), tonumber(row.cr2_hipepic_ds12684), tonumber(row.cr2_hmf_ds13368), tonumber(row.cr2_hmvec_dblad_ds13337), tonumber(row.cr2_hmvec_dblneo_ds13242), tonumber(row.cr2_hmvec_dlyneo_ds13150), tonumber(row.cr2_hmvec_lbl_ds13372), tonumber(row.cr2_hmvec_lly_ds13185), tonumber(row.cr2_hpaf_ds13411), tonumber(row.cr2_hpdlf_ds13573), tonumber(row.cr2_hpf_ds13390), tonumber(row.cr2_hrce_ds10666), tonumber(row.cr2_hsmm_ds14426), tonumber(row.cr2_hth17_ds11039), tonumber(row.cr2_hth1_ds7840), tonumber(row.cr2_hth2ds17597), tonumber(row.cr2_htr_ds14702), tonumber(row.cr2_huvec_ds10060), tonumber(row.cr2_hvmf_ds13981), tonumber(row.cr2_imr90_ds13219), tonumber(row.cr2_ips_19_11_ds15153), tonumber(row.cr2_ith1_ds18018), tonumber(row.cr2_ith2_ds17603), tonumber(row.cr2_k562_ds9767), tonumber(row.cr2_lhcn_m2_ds20548), tonumber(row.cr2_m059j_ds20493), tonumber(row.cr2_mesendoderm_ds19310), tonumber(row.cr2_msc_ds21042), tonumber(row.cr2_nb4_ds12543), tonumber(row.cr2_nha_ds12800), tonumber(row.cr2_nhdf_ad_ds12863), tonumber(row.cr2_nhdf_neo_ds11923), tonumber(row.cr2_nhlf_ds12829), tonumber(row.cr2_psoas_muscle_ds20325), tonumber(row.cr2_rpmi_7951_ds20909), tonumber(row.cr2_saec_ds10518), tonumber(row.cr2_skin_fibroblasts_ds18224), tonumber(row.cr2_skin_keratinocytes_ds18692), tonumber(row.cr2_skin_melanocytes_ds18590), tonumber(row.cr2_skmc_ds11949), tonumber(row.cr2_sknsh_ds8482), tonumber(row.cr2_small_bowel_mucosa_ds20770), tonumber(row.cr2_t_47d_ds19794), tonumber(row.cr2_trophoblast_ds19317), tonumber(row.cr2_vhmec_ds18406), 1 });
	  
	  -- WE DO NOT SELECT "row.source"

	  row = cur:fetch(row, "a");
	  i = i + 1;
	end

	length = #dnaseDataTableTrue;
	lengthTrues = #dnaseDataTableTrue;
	width = (#dnaseDataTableTrue[1])[1];

	print('length = #dnaseDataTableTrue '..comma_value(length));
	-- print('width = (#dnaseDataTableTrue[1])[1] '..width);

if flagNoLimits==true then tuple_limit = length end


	printTime(timeStart, "first Sql execution duration");
	timeSecondSqlExecution = os.time()

	
	-- for i=1,tuple_limit do
	--   io.write('\n dnaseDataTableTrue['..i..']: ');
	--   for j=1,width do
	--   	io.write(' '..dnaseDataTableTrue[i][j]);
	--   end
	--   io.write(';\n');
	-- end io.write("\n");

	-- READING OF THE NonInteractions TABLE IN THE dnaseDataTableFalse ARRAY
	-- CREATION OF THE FINAL dnaseDataTable ARRAY WITH ONE TRUE FOLLOWED BY ONE FALSE AND SO ON
	-- RETURN OF THIS dnaseDataTable OBJECT

	-- -- FALSE INTERACTIONS in the dnaseDataTableFalse 


	sql_query = " SELECT ".. chrNumber  .." AS name, crp1.id_region AS crp1_id_region, crp2.id_region AS crp2_id_region, crp1.a549_ds14289 AS crp1_a549_ds14289,  crp1.ag10803_ds12374 AS crp1_ag10803_ds12374,  crp1.aoaf_ds13513 AS crp1_aoaf_ds13513,  crp1.cd14_ds17215 AS crp1_cd14_ds17215,  crp1.cd19_ds17186 AS crp1_cd19_ds17186,  crp1.cd20_ds18208 AS crp1_cd20_ds18208,  crp1.cd34_ds12274 AS crp1_cd34_ds12274,  crp1.cd3_cordblood_ds17706 AS crp1_cd3_cordblood_ds17706,  crp1.cd3_ds17198 AS crp1_cd3_ds17198,  crp1.cd4_ds17212 AS crp1_cd4_ds17212,  crp1.cd4pos_n_ds14108 AS crp1_cd4pos_n_ds14108,  crp1.cd56_ds17189 AS crp1_cd56_ds17189,  crp1.cd8_ds17203 AS crp1_cd8_ds17203,  crp1.fbrain_ds11872 AS crp1_fbrain_ds11872,  crp1.fheart_ds12531 AS crp1_fheart_ds12531,  crp1.fintestine_lg_ds17313 AS crp1_fintestine_lg_ds17313,  crp1.fkidney_ds20786 AS crp1_fkidney_ds20786,  crp1.flung_ds14724 AS crp1_flung_ds14724,  crp1.fmuscle_leg_ds20239 AS crp1_fmuscle_leg_ds20239,  crp1.fplacenta_ds20346 AS crp1_fplacenta_ds20346,  crp1.fskin_fibro_leg_r_quad_ds19943 AS crp1_fskin_fibro_leg_r_quad_ds19943,  crp1.fspinal_cord_ds20351 AS crp1_fspinal_cord_ds20351,  crp1.fstomach_ds17878 AS crp1_fstomach_ds17878,  crp1.fthymus_ds20341 AS crp1_fthymus_ds20341,  crp1.gm06990_ds7748 AS crp1_gm06990_ds7748,  crp1.gm12865_ds12436 AS crp1_gm12865_ds12436,  crp1.haepic_ds12663 AS crp1_haepic_ds12663,  crp1.hah_ds15192 AS crp1_hah_ds15192,  crp1.hasp_ds14790 AS crp1_hasp_ds14790,  crp1.hcfaa_ds13480 AS crp1_hcfaa_ds13480,  crp1.hcf_ds12501 AS crp1_hcf_ds12501,  crp1.hcm_ds12599 AS crp1_hcm_ds12599,  crp1.hcpepic_ds12447 AS crp1_hcpepic_ds12447,  crp1.heepic_ds12763 AS crp1_heepic_ds12763,  crp1.hepg2_ds7764 AS crp1_hepg2_ds7764,  crp1.hesct0_ds11909 AS crp1_hesct0_ds11909,  crp1.hff_ds15115 AS crp1_hff_ds15115,  crp1.hgf_ds11752 AS crp1_hgf_ds11752,  crp1.hipepic_ds12684 AS crp1_hipepic_ds12684,  crp1.hmf_ds13368 AS crp1_hmf_ds13368,  crp1.hmvec_dblad_ds13337 AS crp1_hmvec_dblad_ds13337,  crp1.hmvec_dblneo_ds13242 AS crp1_hmvec_dblneo_ds13242,  crp1.hmvec_dlyneo_ds13150 AS crp1_hmvec_dlyneo_ds13150,  crp1.hmvec_lbl_ds13372 AS crp1_hmvec_lbl_ds13372,  crp1.hmvec_lly_ds13185 AS crp1_hmvec_lly_ds13185,  crp1.hpaf_ds13411 AS crp1_hpaf_ds13411,  crp1.hpdlf_ds13573 AS crp1_hpdlf_ds13573,  crp1.hpf_ds13390 AS crp1_hpf_ds13390,  crp1.hrce_ds10666 AS crp1_hrce_ds10666,  crp1.hsmm_ds14426 AS crp1_hsmm_ds14426,  crp1.hth17_ds11039 AS crp1_hth17_ds11039,  crp1.hth1_ds7840 AS crp1_hth1_ds7840,  crp1.hth2ds17597 AS crp1_hth2ds17597,  crp1.htr_ds14702 AS crp1_htr_ds14702,  crp1.huvec_ds10060 AS crp1_huvec_ds10060,  crp1.hvmf_ds13981 AS crp1_hvmf_ds13981,  crp1.imr90_ds13219 AS crp1_imr90_ds13219,  crp1.ips_19_11_ds15153 AS crp1_ips_19_11_ds15153,  crp1.ith1_ds18018 AS crp1_ith1_ds18018,  crp1.ith2_ds17603 AS crp1_ith2_ds17603,  crp1.k562_ds9767 AS crp1_k562_ds9767,  crp1.lhcn_m2_ds20548 AS crp1_lhcn_m2_ds20548,  crp1.m059j_ds20493 AS crp1_m059j_ds20493,  crp1.mesendoderm_ds19310 AS crp1_mesendoderm_ds19310,  crp1.msc_ds21042 AS crp1_msc_ds21042,  crp1.nb4_ds12543 AS crp1_nb4_ds12543,  crp1.nha_ds12800 AS crp1_nha_ds12800,  crp1.nhdf_ad_ds12863 AS crp1_nhdf_ad_ds12863,  crp1.nhdf_neo_ds11923 AS crp1_nhdf_neo_ds11923,  crp1.nhlf_ds12829 AS crp1_nhlf_ds12829,  crp1.psoas_muscle_ds20325 AS crp1_psoas_muscle_ds20325,  crp1.rpmi_7951_ds20909 AS crp1_rpmi_7951_ds20909,  crp1.saec_ds10518 AS crp1_saec_ds10518,  crp1.skin_fibroblasts_ds18224 AS crp1_skin_fibroblasts_ds18224,  crp1.skin_keratinocytes_ds18692 AS crp1_skin_keratinocytes_ds18692,  crp1.skin_melanocytes_ds18590 AS crp1_skin_melanocytes_ds18590,  crp1.skmc_ds11949 AS crp1_skmc_ds11949,  crp1.sknsh_ds8482 AS crp1_sknsh_ds8482,  crp1.small_bowel_mucosa_ds20770 AS crp1_small_bowel_mucosa_ds20770,  crp1.t_47d_ds19794 AS crp1_t_47d_ds19794,  crp1.trophoblast_ds19317 AS crp1_trophoblast_ds19317,  crp1.vhmec_ds18406 AS crp1_vhmec_ds18406,  crp2.a549_ds14289 AS crp2_a549_ds14289,  crp2.ag10803_ds12374 AS crp2_ag10803_ds12374,  crp2.aoaf_ds13513 AS crp2_aoaf_ds13513,  crp2.cd14_ds17215 AS crp2_cd14_ds17215,  crp2.cd19_ds17186 ";
	sql_query = sql_query .. " AS crp2_cd19_ds17186, crp2.cd20_ds18208 AS crp2_cd20_ds18208,  crp2.cd34_ds12274 AS crp2_cd34_ds12274,  crp2.cd3_cordblood_ds17706 AS crp2_cd3_cordblood_ds17706,  crp2.cd3_ds17198 AS crp2_cd3_ds17198,  crp2.cd4_ds17212 AS crp2_cd4_ds17212,  crp2.cd4pos_n_ds14108 AS crp2_cd4pos_n_ds14108,  crp2.cd56_ds17189 AS crp2_cd56_ds17189,  crp2.cd8_ds17203 AS crp2_cd8_ds17203,  crp2.fbrain_ds11872 AS crp2_fbrain_ds11872,  crp2.fheart_ds12531 AS crp2_fheart_ds12531,  crp2.fintestine_lg_ds17313 AS crp2_fintestine_lg_ds17313,  crp2.fkidney_ds20786 AS crp2_fkidney_ds20786,  crp2.flung_ds14724 AS crp2_flung_ds14724,  crp2.fmuscle_leg_ds20239 AS crp2_fmuscle_leg_ds20239,  crp2.fplacenta_ds20346 AS crp2_fplacenta_ds20346,  crp2.fskin_fibro_leg_r_quad_ds19943 AS crp2_fskin_fibro_leg_r_quad_ds19943,  crp2.fspinal_cord_ds20351 AS crp2_fspinal_cord_ds20351,  crp2.fstomach_ds17878 AS crp2_fstomach_ds17878,  crp2.fthymus_ds20341 AS crp2_fthymus_ds20341,  crp2.gm06990_ds7748 AS crp2_gm06990_ds7748,  crp2.gm12865_ds12436 AS crp2_gm12865_ds12436,  crp2.haepic_ds12663 AS crp2_haepic_ds12663,  crp2.hah_ds15192 AS crp2_hah_ds15192,  crp2.hasp_ds14790 AS crp2_hasp_ds14790,  crp2.hcfaa_ds13480 AS crp2_hcfaa_ds13480,  crp2.hcf_ds12501 AS crp2_hcf_ds12501,  crp2.hcm_ds12599 AS crp2_hcm_ds12599,  crp2.hcpepic_ds12447 AS crp2_hcpepic_ds12447,  crp2.heepic_ds12763 AS crp2_heepic_ds12763,  crp2.hepg2_ds7764 AS crp2_hepg2_ds7764,  crp2.hesct0_ds11909 AS crp2_hesct0_ds11909,  crp2.hff_ds15115 AS crp2_hff_ds15115,  crp2.hgf_ds11752 AS crp2_hgf_ds11752,  crp2.hipepic_ds12684 AS crp2_hipepic_ds12684,  crp2.hmf_ds13368 AS crp2_hmf_ds13368,  crp2.hmvec_dblad_ds13337 AS crp2_hmvec_dblad_ds13337,  crp2.hmvec_dblneo_ds13242 AS crp2_hmvec_dblneo_ds13242,  crp2.hmvec_dlyneo_ds13150 AS crp2_hmvec_dlyneo_ds13150,  crp2.hmvec_lbl_ds13372 AS crp2_hmvec_lbl_ds13372,  crp2.hmvec_lly_ds13185 AS crp2_hmvec_lly_ds13185,  crp2.hpaf_ds13411 AS crp2_hpaf_ds13411,  crp2.hpdlf_ds13573 AS crp2_hpdlf_ds13573,  crp2.hpf_ds13390 AS crp2_hpf_ds13390,  crp2.hrce_ds10666 AS crp2_hrce_ds10666,  crp2.hsmm_ds14426 AS crp2_hsmm_ds14426,  crp2.hth17_ds11039 AS crp2_hth17_ds11039,  crp2.hth1_ds7840 AS crp2_hth1_ds7840,  crp2.hth2ds17597 AS crp2_hth2ds17597,  crp2.htr_ds14702 AS crp2_htr_ds14702,  crp2.huvec_ds10060 AS crp2_huvec_ds10060,  crp2.hvmf_ds13981 AS crp2_hvmf_ds13981,  crp2.imr90_ds13219 AS crp2_imr90_ds13219,  crp2.ips_19_11_ds15153 AS crp2_ips_19_11_ds15153,  crp2.ith1_ds18018 AS crp2_ith1_ds18018,  crp2.ith2_ds17603 AS crp2_ith2_ds17603,  crp2.k562_ds9767 AS crp2_k562_ds9767,  crp2.lhcn_m2_ds20548 AS crp2_lhcn_m2_ds20548,  crp2.m059j_ds20493 AS crp2_m059j_ds20493,  crp2.mesendoderm_ds19310 AS crp2_mesendoderm_ds19310,  crp2.msc_ds21042 AS crp2_msc_ds21042,  crp2.nb4_ds12543 AS crp2_nb4_ds12543,  crp2.nha_ds12800 AS crp2_nha_ds12800,  crp2.nhdf_ad_ds12863 AS crp2_nhdf_ad_ds12863,  crp2.nhdf_neo_ds11923 AS crp2_nhdf_neo_ds11923,  crp2.nhlf_ds12829 AS crp2_nhlf_ds12829,  crp2.psoas_muscle_ds20325 AS crp2_psoas_muscle_ds20325,  crp2.rpmi_7951_ds20909 AS crp2_rpmi_7951_ds20909,  crp2.saec_ds10518 AS crp2_saec_ds10518,  crp2.skin_fibroblasts_ds18224 AS crp2_skin_fibroblasts_ds18224,  crp2.skin_keratinocytes_ds18692 AS crp2_skin_keratinocytes_ds18692,  crp2.skin_melanocytes_ds18590 AS crp2_skin_melanocytes_ds18590,  crp2.skmc_ds11949 AS crp2_skmc_ds11949,  crp2.sknsh_ds8482 AS crp2_sknsh_ds8482,  crp2.small_bowel_mucosa_ds20770 AS crp2_small_bowel_mucosa_ds20770,  crp2.t_47d_ds19794 AS crp2_t_47d_ds19794,  crp2.trophoblast_ds19317 AS crp2_trophoblast_ds19317,  crp2.vhmec_ds18406  AS crp2_vhmec_ds18406 " ..
	 " FROM chromregions AS cr1 " ..
	 " CROSS JOIN chromregions AS cr2 " ..
	 " JOIN chromosomes AS c " ..
	 " ON (c.id_chr=cr1.id_chr AND c.id_chr=cr2.id_chr) " ..
	 " JOIN chromregionprofiles AS crp1 " ..
	 " ON crp1.id_region = cr1.id_region " ..
	 " JOIN chromregionprofiles AS crp2 " ..
	 " ON crp2.id_region = cr2.id_region " ..
	 " WHERE cr1.id_region <> cr2.id_region " ..
	 " AND NOT EXISTS ( " ..
	 "   SELECT 1 " ..
	 "   FROM trueinteractions AS tix " ..
	 "   WHERE cr1.id_region=tix.id_region1 " ..
	 "   AND cr2.id_region=tix.id_region2 " ..
	 " ) " ..
	 " AND c.name='"..chromSel.. "'" ..
	 " AND (cr1.chrend - cr2.chrstart <"..locus_position_limit.. ") " ..
	 " AND (cr2.chrend - cr1.chrstart <"..locus_position_limit.. ") " .. " ORDER BY random() "
	-- FORMER: " ORDER BY crp1_id_region, crp2_id_region "
	
	
	if balancedFlag==false and original_tuple_limit ~= -1 then sql_query = sql_query ..	 " LIMIT "..tonumber(original_tuple_limit*2) end

	sql_query = sql_query .. ";"
	
	--   print("\tsql_query: \n"..sql_query);  io.flush();
	 
	-- retrieve a cursor
	 cur = assert (openGlobalDbConnection():execute(string.format([[%s]], sql_query)));
	  
	 
	-- print all rows, the rows will be indexed by field names
	row = cur:fetch ({}, "a");
	-- print(string.format("first_chrname-first_chrstart-first_chrend\tsecond_chrname-second_chrstart-second_chrend\n"))
	 
	
	-- print("\t\tafter cur:fetch()"); io.flush();
	
	 
	 dnaseDataTableFalse = {}
	 i = 1;

	 
	 while row do
	   
	   -- dnaseDataTableFalse[i] = {};
	   
	   dnaseDataTableFalse[i] = torch.Tensor({tonumber(row.name), tonumber(row.crp1_id_region), tonumber(row.crp2_id_region), tonumber(row.crp1_a549_ds14289), tonumber(row.crp1_ag10803_ds12374), tonumber(row.crp1_aoaf_ds13513), tonumber(row.crp1_cd14_ds17215), tonumber(row.crp1_cd19_ds17186), tonumber(row.crp1_cd20_ds18208), tonumber(row.crp1_cd34_ds12274), tonumber(row.crp1_cd3_cordblood_ds17706), tonumber(row.crp1_cd3_ds17198), tonumber(row.crp1_cd4_ds17212), tonumber(row.crp1_cd4pos_n_ds14108), tonumber(row.crp1_cd56_ds17189), tonumber(row.crp1_cd8_ds17203), tonumber(row.crp1_fbrain_ds11872), tonumber(row.crp1_fheart_ds12531), tonumber(row.crp1_fintestine_lg_ds17313), tonumber(row.crp1_fkidney_ds20786), tonumber(row.crp1_flung_ds14724), tonumber(row.crp1_fmuscle_leg_ds20239), tonumber(row.crp1_fplacenta_ds20346), tonumber(row.crp1_fskin_fibro_leg_r_quad_ds19943), tonumber(row.crp1_fspinal_cord_ds20351), tonumber(row.crp1_fstomach_ds17878), tonumber(row.crp1_fthymus_ds20341), tonumber(row.crp1_gm06990_ds7748), tonumber(row.crp1_gm12865_ds12436), tonumber(row.crp1_haepic_ds12663), tonumber(row.crp1_hah_ds15192), tonumber(row.crp1_hasp_ds14790), tonumber(row.crp1_hcfaa_ds13480), tonumber(row.crp1_hcf_ds12501), tonumber(row.crp1_hcm_ds12599), tonumber(row.crp1_hcpepic_ds12447), tonumber(row.crp1_heepic_ds12763), tonumber(row.crp1_hepg2_ds7764), tonumber(row.crp1_hesct0_ds11909), tonumber(row.crp1_hff_ds15115), tonumber(row.crp1_hgf_ds11752), tonumber(row.crp1_hipepic_ds12684), tonumber(row.crp1_hmf_ds13368), tonumber(row.crp1_hmvec_dblad_ds13337), tonumber(row.crp1_hmvec_dblneo_ds13242), tonumber(row.crp1_hmvec_dlyneo_ds13150), tonumber(row.crp1_hmvec_lbl_ds13372), tonumber(row.crp1_hmvec_lly_ds13185), tonumber(row.crp1_hpaf_ds13411), tonumber(row.crp1_hpdlf_ds13573), tonumber(row.crp1_hpf_ds13390), tonumber(row.crp1_hrce_ds10666), tonumber(row.crp1_hsmm_ds14426), tonumber(row.crp1_hth17_ds11039), tonumber(row.crp1_hth1_ds7840), tonumber(row.crp1_hth2ds17597), tonumber(row.crp1_htr_ds14702), tonumber(row.crp1_huvec_ds10060), tonumber(row.crp1_hvmf_ds13981), tonumber(row.crp1_imr90_ds13219), tonumber(row.crp1_ips_19_11_ds15153), tonumber(row.crp1_ith1_ds18018), tonumber(row.crp1_ith2_ds17603), tonumber(row.crp1_k562_ds9767), tonumber(row.crp1_lhcn_m2_ds20548), tonumber(row.crp1_m059j_ds20493), tonumber(row.crp1_mesendoderm_ds19310), tonumber(row.crp1_msc_ds21042), tonumber(row.crp1_nb4_ds12543), tonumber(row.crp1_nha_ds12800), tonumber(row.crp1_nhdf_ad_ds12863), tonumber(row.crp1_nhdf_neo_ds11923), tonumber(row.crp1_nhlf_ds12829), tonumber(row.crp1_psoas_muscle_ds20325), tonumber(row.crp1_rpmi_7951_ds20909), tonumber(row.crp1_saec_ds10518), tonumber(row.crp1_skin_fibroblasts_ds18224), tonumber(row.crp1_skin_keratinocytes_ds18692), tonumber(row.crp1_skin_melanocytes_ds18590), tonumber(row.crp1_skmc_ds11949), tonumber(row.crp1_sknsh_ds8482), tonumber(row.crp1_small_bowel_mucosa_ds20770), tonumber(row.crp1_t_47d_ds19794), tonumber(row.crp1_trophoblast_ds19317), tonumber(row.crp1_vhmec_ds18406),
	     tonumber(row.crp2_a549_ds14289), tonumber(row.crp2_ag10803_ds12374), tonumber(row.crp2_aoaf_ds13513), tonumber(row.crp2_cd14_ds17215), tonumber(row.crp2_cd19_ds17186), tonumber(row.crp2_cd20_ds18208), tonumber(row.crp2_cd34_ds12274), tonumber(row.crp2_cd3_cordblood_ds17706), tonumber(row.crp2_cd3_ds17198), tonumber(row.crp2_cd4_ds17212), tonumber(row.crp2_cd4pos_n_ds14108), tonumber(row.crp2_cd56_ds17189), tonumber(row.crp2_cd8_ds17203), tonumber(row.crp2_fbrain_ds11872), tonumber(row.crp2_fheart_ds12531), tonumber(row.crp2_fintestine_lg_ds17313), tonumber(row.crp2_fkidney_ds20786), tonumber(row.crp2_flung_ds14724), tonumber(row.crp2_fmuscle_leg_ds20239), tonumber(row.crp2_fplacenta_ds20346), tonumber(row.crp2_fskin_fibro_leg_r_quad_ds19943), tonumber(row.crp2_fspinal_cord_ds20351), tonumber(row.crp2_fstomach_ds17878), tonumber(row.crp2_fthymus_ds20341), tonumber(row.crp2_gm06990_ds7748), tonumber(row.crp2_gm12865_ds12436), tonumber(row.crp2_haepic_ds12663), tonumber(row.crp2_hah_ds15192), tonumber(row.crp2_hasp_ds14790), tonumber(row.crp2_hcfaa_ds13480), tonumber(row.crp2_hcf_ds12501), tonumber(row.crp2_hcm_ds12599), tonumber(row.crp2_hcpepic_ds12447), tonumber(row.crp2_heepic_ds12763), tonumber(row.crp2_hepg2_ds7764), tonumber(row.crp2_hesct0_ds11909), tonumber(row.crp2_hff_ds15115), tonumber(row.crp2_hgf_ds11752), tonumber(row.crp2_hipepic_ds12684), tonumber(row.crp2_hmf_ds13368), tonumber(row.crp2_hmvec_dblad_ds13337), tonumber(row.crp2_hmvec_dblneo_ds13242), tonumber(row.crp2_hmvec_dlyneo_ds13150), tonumber(row.crp2_hmvec_lbl_ds13372), tonumber(row.crp2_hmvec_lly_ds13185), tonumber(row.crp2_hpaf_ds13411), tonumber(row.crp2_hpdlf_ds13573), tonumber(row.crp2_hpf_ds13390), tonumber(row.crp2_hrce_ds10666), tonumber(row.crp2_hsmm_ds14426), tonumber(row.crp2_hth17_ds11039), tonumber(row.crp2_hth1_ds7840), tonumber(row.crp2_hth2ds17597), tonumber(row.crp2_htr_ds14702), tonumber(row.crp2_huvec_ds10060), tonumber(row.crp2_hvmf_ds13981), tonumber(row.crp2_imr90_ds13219), tonumber(row.crp2_ips_19_11_ds15153), tonumber(row.crp2_ith1_ds18018), tonumber(row.crp2_ith2_ds17603), tonumber(row.crp2_k562_ds9767), tonumber(row.crp2_lhcn_m2_ds20548), tonumber(row.crp2_m059j_ds20493), tonumber(row.crp2_mesendoderm_ds19310), tonumber(row.crp2_msc_ds21042), tonumber(row.crp2_nb4_ds12543), tonumber(row.crp2_nha_ds12800), tonumber(row.crp2_nhdf_ad_ds12863), tonumber(row.crp2_nhdf_neo_ds11923), tonumber(row.crp2_nhlf_ds12829), tonumber(row.crp2_psoas_muscle_ds20325), tonumber(row.crp2_rpmi_7951_ds20909), tonumber(row.crp2_saec_ds10518), tonumber(row.crp2_skin_fibroblasts_ds18224), tonumber(row.crp2_skin_keratinocytes_ds18692), tonumber(row.crp2_skin_melanocytes_ds18590), tonumber(row.crp2_skmc_ds11949), tonumber(row.crp2_sknsh_ds8482), tonumber(row.crp2_small_bowel_mucosa_ds20770), tonumber(row.crp2_t_47d_ds19794), tonumber(row.crp2_trophoblast_ds19317), tonumber(row.crp2_vhmec_ds18406), 0 });
	   
	--   -- WE DO NOT SELECT "row.source"
	 
	   io.write("i="..i.."\t"); io.flush();
	   
	   row = cur:fetch(row, "a");
	   i = i + 1;
	   
	 end
	 
         printTime(timeSecondSqlExecution, "second Sql execution duration");

	 
	 length = #dnaseDataTableFalse;
	 width = (#dnaseDataTableFalse[1])[1];
	 
	 
	print('length = #dnaseDataTableFalse '..comma_value(length));
	-- print('width = (#dnaseDataTableFalse[1])[1] '..width..'\n');
	 
	-- for i=1,tuple_limit do
	  -- io.write('\n dnaseDataTableFalse['..i..']: ');
	  -- for j=1,width do
	--	io.write('[i='..i..'][j='..j..'] '); io:flush();
	--   	io.write(' '..dnaseDataTableFalse[i][j]..'\n');
	--   end
	--   io.write(';\n');
	-- end
	-- io.write("\n");

	 dnaseDataTable = {};

	  i=1;
	  j=1;
	 for u=1,original_tuple_limit*2 do
	  dnaseDataTable[u] = {};
		if u%2==0 then
		   	dnaseDataTable[u] = (dnaseDataTableTrue[i]);			
			i = i + 1;
		else
		   	dnaseDataTable[u] = (dnaseDataTableFalse[j]);
			j = j + 1;
		end
	 end

	 t=1
	 if balancedFlag==false then
	  for t=tuple_limit*2+1,tuple_limit*2+#dnaseDataTableFalse do
		dnaseDataTable[t] = {};
		dnaseDataTable[t] = (dnaseDataTableFalse[j]);
		j = j + 1;
	    end
	 end
	 
-- 	for i=1,#dnaseDataTable do
-- 	  io.write('\n dnaseDataTable['..i..']: ');
-- 	  for j=1,width do
-- 	  	io.write(' '..dnaseDataTable[i][j]); io:flush();
-- 	  end
-- 	  io.write(';\n');
-- 	end
-- 	io.write("\n");

	 
	dnaseDataTable_length = #dnaseDataTable;
	dnaseDataTable_width = (#dnaseDataTable[1])[1];
	 
	 
	print('dnaseDataTable_length = #dnaseDataTable '..comma_value(dnaseDataTable_length));
	print('dnaseDataTable_width = (#dnaseDataTable[1])[1 '..comma_value(dnaseDataTable_width)..'\n');

	-- close everything
	cur:close(); -- already closed because all the result set was consumed
	-- closeGlobalDbConnection()
	
	printTime(timeStart, "PostgreSQL data reading")

	return {lengthTrues, dnaseDataTable};

end

--[[
-- Row former sql configuration
torch.Tensor({tonumber(row.name), tonumber(row.crp1_id_region), tonumber(row.crp2_id_region), tonumber(row.crp1_a549_ds14289), tonumber(row.crp1_ag10803_ds12374), tonumber(row.crp1_aoaf_ds13513), tonumber(row.crp1_cd14_ds17215), tonumber(row.crp1_cd19_ds17186), tonumber(row.crp1_cd20_ds18208), tonumber(row.crp1_cd34_ds12274), tonumber(row.crp1_cd3_cordblood_ds17706), tonumber(row.crp1_cd3_ds17198), tonumber(row.crp1_cd4_ds17212), tonumber(row.crp1_cd4pos_n_ds14108), tonumber(row.crp1_cd56_ds17189), tonumber(row.crp1_cd8_ds17203), tonumber(row.crp1_fbrain_ds11872), tonumber(row.crp1_fheart_ds12531), tonumber(row.crp1_fintestine_lg_ds17313), tonumber(row.crp1_fkidney_ds20786), tonumber(row.crp1_flung_ds14724), tonumber(row.crp1_fmuscle_leg_ds20239), tonumber(row.crp1_fplacenta_ds20346), tonumber(row.crp1_fskin_fibro_leg_r_quad_ds19943), tonumber(row.crp1_fspinal_cord_ds20351), tonumber(row.crp1_fstomach_ds17878), tonumber(row.crp1_fthymus_ds20341), tonumber(row.crp1_gm06990_ds7748), tonumber(row.crp1_gm12865_ds12436), tonumber(row.crp1_haepic_ds12663), tonumber(row.crp1_hah_ds15192), tonumber(row.crp1_hasp_ds14790), tonumber(row.crp1_hcfaa_ds13480), tonumber(row.crp1_hcf_ds12501), tonumber(row.crp1_hcm_ds12599), tonumber(row.crp1_hcpepic_ds12447), tonumber(row.crp1_heepic_ds12763), tonumber(row.crp1_hepg2_ds7764), tonumber(row.crp1_hesct0_ds11909), tonumber(row.crp1_hff_ds15115), tonumber(row.crp1_hgf_ds11752), tonumber(row.crp1_hipepic_ds12684), tonumber(row.crp1_hmf_ds13368), tonumber(row.crp1_hmvec_dblad_ds13337), tonumber(row.crp1_hmvec_dblneo_ds13242), tonumber(row.crp1_hmvec_dlyneo_ds13150), tonumber(row.crp1_hmvec_lbl_ds13372), tonumber(row.crp1_hmvec_lly_ds13185), tonumber(row.crp1_hpaf_ds13411), tonumber(row.crp1_hpdlf_ds13573), tonumber(row.crp1_hpf_ds13390), tonumber(row.crp1_hrce_ds10666), tonumber(row.crp1_hsmm_ds14426), tonumber(row.crp1_hth17_ds11039), tonumber(row.crp1_hth1_ds7840), tonumber(row.crp1_hth2ds17597), tonumber(row.crp1_htr_ds14702), tonumber(row.crp1_huvec_ds10060), tonumber(row.crp1_hvmf_ds13981), tonumber(row.crp1_imr90_ds13219), tonumber(row.crp1_ips_19_11_ds15153), tonumber(row.crp1_ith1_ds18018), tonumber(row.crp1_ith2_ds17603), tonumber(row.crp1_k562_ds9767), tonumber(row.crp1_lhcn_m2_ds20548), tonumber(row.crp1_m059j_ds20493), tonumber(row.crp1_mesendoderm_ds19310), tonumber(row.crp1_msc_ds21042), tonumber(row.crp1_nb4_ds12543), tonumber(row.crp1_nha_ds12800), tonumber(row.crp1_nhdf_ad_ds12863), tonumber(row.crp1_nhdf_neo_ds11923), tonumber(row.crp1_nhlf_ds12829), tonumber(row.crp1_psoas_muscle_ds20325), tonumber(row.crp1_rpmi_7951_ds20909), tonumber(row.crp1_saec_ds10518), tonumber(row.crp1_skin_fibroblasts_ds18224), tonumber(row.crp1_skin_keratinocytes_ds18692), tonumber(row.crp1_skin_melanocytes_ds18590), tonumber(row.crp1_skmc_ds11949), tonumber(row.crp1_sknsh_ds8482), tonumber(row.crp1_small_bowel_mucosa_ds20770), tonumber(row.crp1_t_47d_ds19794), tonumber(row.crp1_trophoblast_ds19317), tonumber(row.crp1_vhmec_ds18406), tonumber(row.crp2_a549_ds14289), tonumber(row.crp2_ag10803_ds12374), tonumber(row.crp2_aoaf_ds13513), tonumber(row.crp2_cd14_ds17215), tonumber(row.crp2_cd19_ds17186), tonumber(row.crp2_cd20_ds18208), tonumber(row.crp2_cd34_ds12274), tonumber(row.crp2_cd3_cordblood_ds17706), tonumber(row.crp2_cd3_ds17198), tonumber(row.crp2_cd4_ds17212), tonumber(row.crp2_cd4pos_n_ds14108), tonumber(row.crp2_cd56_ds17189), tonumber(row.crp2_cd8_ds17203), tonumber(row.crp2_fbrain_ds11872), tonumber(row.crp2_fheart_ds12531), tonumber(row.crp2_fintestine_lg_ds17313), tonumber(row.crp2_fkidney_ds20786), tonumber(row.crp2_flung_ds14724), tonumber(row.crp2_fmuscle_leg_ds20239), tonumber(row.crp2_fplacenta_ds20346), tonumber(row.crp2_fskin_fibro_leg_r_quad_ds19943), tonumber(row.crp2_fspinal_cord_ds20351),  tonumber(row.crp2_fstomach_ds17878), tonumber(row.crp2_fthymus_ds20341), tonumber(row.crp2_gm06990_ds7748), tonumber(row.crp2_gm12865_ds12436), tonumber(row.crp2_haepic_ds12663), tonumber(row.crp2_hah_ds15192), tonumber(row.crp2_hasp_ds14790), tonumber(row.crp2_hcfaa_ds13480), tonumber(
row.crp2_hcf_ds12501), tonumber(row.crp2_hcm_ds12599), tonumber(row.crp2_hcpepic_ds12447), tonumber(row.crp2_heepic_ds12763), tonumber(row.crp2_hepg2_ds7764), tonumber(row.crp2_hesct0_ds11909), tonumber(row.crp2_hff_ds15115), tonumber(row.crp2_hgf_ds11752), tonumber(row.crp2_hipepic_ds12684), tonumber(row.crp2_hmf_ds13368), tonumber(row.crp2_hmvec_dblad_ds13337), tonumber(row.crp2_hmvec_dblneo_ds13242), tonumber(row.crp2_hmvec_dlyneo_ds13150), tonumber(row.crp2_hmvec_lbl_ds13372), tonumber(row.crp2_hmvec_lly_ds13185), tonumber(row.crp2_hpaf_ds13411), tonumber(row.crp2_hpdlf_ds13573), tonumber(row.crp2_hpf_ds13390), tonumber(row.crp2_hrce_ds10666), tonumber(row.crp2_hsmm_ds14426), tonumber(row.crp2_hth17_ds11039), tonumber(row.crp2_hth1_ds7840),  tonumber(row.crp2_hth2ds17597), tonumber(row.crp2_htr_ds14702), tonumber(row.crp2_huvec_ds10060), tonumber(row.crp2_hvmf_ds13981), tonumber(row.crp2_imr90_ds13219), tonumber(row.crp2_ips_19_11_ds15153), tonumber(row.crp2_ith1_ds18018), tonumber(row.crp2_ith2_ds17603), tonumber(row.crp2_k562_ds9767),  tonumber(row.crp2_lhcn_m2_ds20548), tonumber(row.crp2_m059j_ds20493), tonumber(row.crp2_mesendoderm_ds19310), tonumber(row.crp2_msc_ds21042), tonumber(row.crp2_nb4_ds12543), tonumber(row.crp2_nha_ds12800),  tonumber(row.crp2_nhdf_ad_ds12863), tonumber(row.crp2_nhdf_neo_ds11923), tonumber(row.crp2_nhlf_ds12829), tonumber(row.crp2_psoas_muscle_ds20325), tonumber(row.crp2_rpmi_7951_ds20909), tonumber(row.crp2_saec_ds10518), tonumber(row.crp2_skin_fibroblasts_ds18224), tonumber(row.crp2_skin_keratinocytes_ds18692),  tonumber(row.crp2_skin_melanocytes_ds18590), tonumber(row.crp2_skmc_ds11949), tonumber(row.crp2_sknsh_ds8482), tonumber(row.crp2_small_bowel_mucosa_ds20770), tonumber(row.crp2_t_47d_ds19794), tonumber(row.crp2_trophoblast_ds19317), tonumber(row.crp2_vhmec_ds18406), 0 });]]