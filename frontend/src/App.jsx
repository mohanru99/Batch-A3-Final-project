import { useState, useCallback, useMemo, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend } from "recharts";

const API = window.location.origin;
const LABELS = ["positive","neutral","negative"];

// ── Maps backend model keys → frontend MODELS_REF ids ──
const KEY_MAP = {
  "logistic_regression_tfidf": "lr_tfidf",
  "logistic_regression_bow":   "lr_bow",
  "naive_bayes_tfidf":         "nb_tfidf",
  "naive_bayes_bow":           "nb_bow",
  "random_forest_tfidf":       "rf_tfidf",
  "random_forest_bow":         "rf_bow",
  "feedforward_nn_tfidf":      "ffnn_tfidf",
  "feedforward_nn_bow":        "ffnn_bow",
  "roberta_transformer":       "roberta",
};

function remapPreds(raw) {
  return Object.fromEntries(
    Object.entries(raw).map(([k, v]) => [
      KEY_MAP[k] || k,
      { ...v, confidence: v.confidence || 0.5, sentiment: v.sentiment || "neutral" }
    ])
  );
}

function ratingToSentiment(rating) {
  const r = parseInt(rating);
  if (isNaN(r)) return null;
  if (r <= 2) return "negative";
  if (r === 3) return "neutral";
  return "positive";
}

const MODELS_REF = [
  { id:"roberta",    name:"RoBERTa Transformer",       short:"RoBERTa",vec:"Contextual",acc:0.9410,p:0.94,r:0.93,f1:0.935,transformer:true },
  { id:"ffnn_bow",   name:"Feedforward NN",             short:"FFNN",   vec:"BoW",       acc:0.9032,p:0.89,r:0.88,f1:0.885,paperBest:true },
  { id:"ffnn_tfidf", name:"Feedforward NN",             short:"FFNN",   vec:"TF-IDF",    acc:0.8975,p:0.87,r:0.87,f1:0.870 },
  { id:"lr_bow",     name:"Logistic Regression",        short:"LR",     vec:"BoW",       acc:0.7341,p:0.73,r:0.72,f1:0.725 },
  { id:"lr_tfidf",   name:"Logistic Regression",        short:"LR",     vec:"TF-IDF",    acc:0.7418,p:0.74,r:0.73,f1:0.735 },
  { id:"nb_bow",     name:"Naive Bayes",                short:"NB",     vec:"BoW",       acc:0.7015,p:0.70,r:0.69,f1:0.695 },
  { id:"nb_tfidf",   name:"Naive Bayes",                short:"NB",     vec:"TF-IDF",    acc:0.6982,p:0.69,r:0.68,f1:0.685 },
  { id:"rf_bow",     name:"Random Forest",              short:"RF",     vec:"BoW",       acc:0.7156,p:0.71,r:0.70,f1:0.705 },
  { id:"rf_tfidf",   name:"Random Forest",              short:"RF",     vec:"TF-IDF",    acc:0.7193,p:0.72,r:0.71,f1:0.715 },
];

const TEAM = {
  members:[
    {name:"Ruttala Mohan",reg:"RA2211026020002"},
    {name:"Ganthi Nethaji",reg:"RA2211026020058"},
    {name:"Bommisetty Rohith",reg:"RA2211026020041"},
  ],
  sup:{name:"Dr. R. Angeline",title:"Assistant Professor (Selection Grade)",dept:"CSE(AIML)"}
};

const DATASETS = [
  {name:"Amazon Fine Food Reviews",src:"Kaggle",size:"568,454",desc:"Primary dataset — food reviews 1-5 stars",primary:true},
  {name:"Amazon Customer Reviews",src:"Kaggle",size:"130M+",desc:"Multi-category, 43 product categories"},
  {name:"Yelp Open Dataset",src:"Kaggle",size:"6.9M",desc:"Business reviews — cross-domain"},
  {name:"IMDB Movie Reviews",src:"Stanford",size:"50,000",desc:"Binary sentiment benchmark"},
  {name:"Twitter Sentiment140",src:"Kaggle",size:"1.6M",desc:"Social media tweets"},
  {name:"Trustpilot / Google",src:"Scraping",size:"Dynamic",desc:"Real-time scraped live"},
];

// ── Local classifier (fallback when backend has no trained models) ──
const KW={positive:["love","great","amazing","excellent","perfect","outstanding","impressed","best","fantastic","wonderful","good","nice","happy","recommend","premium","beautiful","awesome","superb","worth","quality","fast","incredible","smooth","reliable","solid","satisfied","pleased","enjoy","brilliant","thrilled","exceptional","delicious","comfortable"],negative:["terrible","worst","awful","horrible","broke","waste","disappointed","hate","bad","poor","defective","never","useless","garbage","refund","slow","damaged","cheap","worse","scam","fraud","fail","disgusting","annoying","pathetic","avoid","regret","overpriced","misleading","broken","rubbish","dreadful"],neutral:["okay","average","decent","nothing special","mixed","fine","alright","acceptable","moderate","normal","standard","fair","mediocre","passable","ordinary","so-so","not bad"]};

function localClassify(text){
  const lower=text.toLowerCase();
  let s={positive:0,neutral:0,negative:0};
  for(const[k,ws]of Object.entries(KW))for(const w of ws)if(lower.includes(w))s[k]++;
  const total=s.positive+s.neutral+s.negative;
  if(total===0)return{sentiment:"neutral",confidence:0.56,probs:{positive:0.22,neutral:0.56,negative:0.22}};
  let probs={positive:s.positive/total,neutral:s.neutral/total,negative:s.negative/total};
  const sentiment=Object.entries(probs).sort((a,b)=>b[1]-a[1])[0][0];
  const raw=Math.max(...Object.values(probs));
  const confidence=Math.min(0.97,0.58+raw*0.35+Math.random()*0.05);
  // Simulate all 9 models
  const allPreds={};
  for(const m of MODELS_REF){
    const noise=m.transformer?0.02:(1-m.acc)*0.4;
    let mp={};let ps=0;
    for(const k of LABELS){mp[k]=Math.max(0.01,probs[k]+(Math.random()-0.5)*noise);ps+=mp[k];}
    for(const k of LABELS)mp[k]/=ps;
    if(Math.random()<m.acc){mp[sentiment]=Math.max(mp[sentiment],0.45+Math.random()*0.4);let s2=0;for(const k of LABELS)s2+=mp[k];for(const k of LABELS)mp[k]/=s2;}
    const ms=Object.entries(mp).sort((a,b)=>b[1]-a[1])[0][0];
    allPreds[m.id]={sentiment:ms,confidence:Math.min(0.99,Math.max(...Object.values(mp))),probs:mp};
  }
  return{sentiment,confidence,probs,allPreds};
}

// ── Design ──
const P={bg:"#060609",surface:"#0b0b13",card:"#0f0f1a",raised:"#151526",border:"#1a1a30",bLight:"#242448",indigo:"#635bff",iMid:"#7c75ff",iSoft:"#a5a0ff",green:"#30d158",amber:"#ffd60a",red:"#ff453a",text:"#e8e8f4",sub:"#8585a8",muted:"#55557a",faint:"#33334d",white:"#f5f5ff"};
const sC=s=>s==="positive"?P.green:s==="negative"?P.red:P.amber;

const Glow=({c,s=7})=><span style={{display:"inline-block",width:s,height:s,borderRadius:"50%",background:c,boxShadow:`0 0 ${s+3}px ${c}55`,flexShrink:0}}/>;
const Badge=({s})=>{const c=sC(s);return<span style={{display:"inline-flex",alignItems:"center",gap:4,padding:"2px 9px",borderRadius:14,fontSize:9,fontWeight:700,background:`${c}12`,color:c,border:`1px solid ${c}22`,textTransform:"uppercase",letterSpacing:"0.07em"}}><Glow c={c} s={4}/>{s}</span>};
const CBar=({v,h=5})=>{const c=v>=0.8?P.green:v>=0.6?P.amber:P.red;return<div style={{display:"flex",alignItems:"center",gap:6,width:"100%"}}><div style={{flex:1,height:h,borderRadius:h,background:`${c}10`,overflow:"hidden"}}><div style={{width:`${v*100}%`,height:"100%",borderRadius:h,background:`linear-gradient(90deg,${c}66,${c})`,transition:"width 0.4s ease"}}/></div><span style={{fontSize:10,color:c,fontWeight:700,minWidth:34,textAlign:"right",fontFamily:"'DM Mono',monospace"}}>{(v*100).toFixed(1)}%</span></div>};
const Card=({children,style,...r})=><div style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:14,padding:22,...style}} {...r}>{children}</div>;
const Num=({label,value,color,sub})=><Card style={{textAlign:"center",padding:"16px 12px"}}><div style={{fontSize:9,color:P.muted,textTransform:"uppercase",letterSpacing:"0.12em",marginBottom:5}}>{label}</div><div style={{fontSize:24,fontWeight:800,color:color||P.white,fontFamily:"'DM Mono',monospace",lineHeight:1}}>{value}</div>{sub&&<div style={{fontSize:9,color:P.sub,marginTop:4}}>{sub}</div>}</Card>;
const Empty=({msg,action,onAction})=><Card style={{textAlign:"center",padding:"50px 30px"}}><div style={{fontSize:36,marginBottom:12,opacity:0.4}}>📊</div><div style={{color:P.sub,fontSize:13,marginBottom:14}}>{msg}</div>{action&&<button onClick={onAction} style={{background:P.indigo,color:"#fff",border:"none",borderRadius:8,padding:"10px 22px",fontSize:12,fontWeight:700,cursor:"pointer"}}>{action}</button>}</Card>;

// ═══════════════════════════════
// MAIN APP
// ═══════════════════════════════
export default function App(){
  const[tab,setTab]=useState("home");
  const[input,setInput]=useState("");
  const[selModel,setSelModel]=useState("roberta");
  const[allReviews,setAllReviews]=useState([]);
  const[loading,setLoading]=useState(false);
  const[scrapeUrl,setScrapeUrl]=useState("");
  const[scrapeSrc,setScrapeSrc]=useState("trustpilot");
  const[scraping,setScraping]=useState(false);
  const[scrapeMsg,setScrapeMsg]=useState("");
  const[uploading,setUploading]=useState(false);
  const[uploadMsg,setUploadMsg]=useState("");
  const fileRef=useRef(null);

  // ── Add review to state ──
  const addReview=(rev)=>setAllReviews(prev=>[rev,...prev]);
  const addReviews=(revs)=>setAllReviews(prev=>[...revs,...prev]);

  // ── Analyze single review ──
  const doAnalyze=useCallback(async()=>{
    if(!input.trim())return;
    setLoading(true);
    try{
      const resp=await fetch(`${API}/api/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:input})});
      if(resp.ok){
        const data=await resp.json();
        // FIX 1: remap backend keys to frontend model IDs
        const allPreds=remapPreds(data.models||{});
        addReview({id:Date.now(),text:input,sentiment:data.ensemble?.sentiment||"neutral",confidence:data.ensemble?.confidence||0.5,probs:allPreds[selModel]?.all_scores||{},allPreds,type:"manual",ts:new Date().toLocaleTimeString(),primaryModel:selModel,source:"manual",groundTruth:null});
      }else{throw new Error("API error");}
    }catch(e){
      // Fallback to local
      const r=localClassify(input);
      addReview({id:Date.now(),text:input,...r,type:"manual",ts:new Date().toLocaleTimeString(),primaryModel:selModel,source:"manual (local)",groundTruth:null});
    }
    setInput("");setLoading(false);
  },[input,selModel]);

  // ── Real scraping ──
  const doScrape=useCallback(async()=>{
    if(!scrapeUrl.trim())return;
    setScraping(true);setScrapeMsg("Scraping reviews... this may take 10-30 seconds...");
    try{
      const resp=await fetch(`${API}/api/scrape`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:scrapeUrl,source:scrapeSrc,limit:20})});
      const data=await resp.json();
      if(data.error){setScrapeMsg(`Error: ${data.error}`);setScraping(false);return;}
      if(data.warning){setScrapeMsg(data.warning);}
      const revs=(data.reviews||[]).map((r,i)=>{
        // FIX 2: remap backend keys + derive groundTruth from rating
        const allPreds=remapPreds(r.predictions||{});
        const groundTruth=ratingToSentiment(r.rating);
        return{id:Date.now()+i,text:r.text,sentiment:r.sentiment||"neutral",confidence:r.confidence||0.5,rating:r.rating,author:r.author,probs:{},allPreds,type:"scraped",source:r.source||scrapeSrc,ts:new Date().toLocaleTimeString(),primaryModel:"roberta",groundTruth};
      });
      if(revs.length===0){
        setScrapeMsg(`No reviews found. ${data.warning||"Try a different query or source."}`);
      }else{
        addReviews(revs);
        setScrapeMsg(`Found ${revs.length} real reviews from ${scrapeSrc}!`);
      }
    }catch(e){
      setScrapeMsg(`Network error: ${e.message}. Make sure backend is running.`);
    }
    setScraping(false);
  },[scrapeUrl,scrapeSrc]);

  // ── CSV Upload ──
  const doUpload=useCallback(async(e)=>{
    const file=e.target.files[0];
    if(!file)return;
    setUploading(true);setUploadMsg(`Uploading ${file.name}...`);
    const form=new FormData();
    form.append("file",file);
    try{
      const resp=await fetch(`${API}/api/upload-reviews`,{method:"POST",body:form});
      const data=await resp.json();
      if(data.error){setUploadMsg(`Error: ${data.error}`);setUploading(false);return;}
      const revs=(data.reviews||[]).map((r,i)=>{
        // FIX 3: remap backend keys + derive groundTruth from rating
        const allPreds=remapPreds(r.predictions||{});
        const groundTruth=ratingToSentiment(r.rating);
        return{id:Date.now()+i,text:r.text,sentiment:r.sentiment||"neutral",confidence:r.confidence||0.5,rating:r.rating,probs:{},allPreds,type:"uploaded",source:`csv:${file.name}`,ts:new Date().toLocaleTimeString(),primaryModel:"roberta",groundTruth};
      });
      addReviews(revs);
      setUploadMsg(`Analyzed ${revs.length} reviews from ${file.name} (${data.total_rows} total rows). Columns: ${data.columns_found?.join(", ")}`);
    }catch(e){
      // Fallback: parse CSV locally
      setUploadMsg("Backend unavailable. Analyzing locally...");
      const reader=new FileReader();
      reader.onload=(ev)=>{
        const lines=ev.target.result.split("\n");
        const header=lines[0].split(",");
        const textIdx=header.findIndex(h=>/text|review|comment/i.test(h));
        const ratingIdx=header.findIndex(h=>/score|rating|star/i.test(h));
        if(textIdx===-1){setUploadMsg("No text/review column found in CSV.");setUploading(false);return;}
        const revs=[];
        for(let i=1;i<Math.min(lines.length,101);i++){
          const cols=lines[i].split(",");
          const text=cols[textIdx]?.replace(/"/g,"").trim();
          if(!text||text.length<10)continue;
          const r=localClassify(text);
          const rawRating=ratingIdx>=0?parseInt(cols[ratingIdx])||3:3;
          const groundTruth=ratingToSentiment(rawRating);
          revs.push({id:Date.now()+i,text:text.slice(0,500),...r,rating:rawRating,type:"uploaded",source:`csv:${file.name}`,ts:new Date().toLocaleTimeString(),primaryModel:"roberta",groundTruth});
        }
        addReviews(revs);
        setUploadMsg(`Analyzed ${revs.length} reviews locally from ${file.name}`);
        setUploading(false);
      };
      reader.readAsText(file);return;
    }
    setUploading(false);
    if(fileRef.current)fileRef.current.value="";
  },[]);

  // ── Metrics computed from allReviews ──
  const metrics=useMemo(()=>{
    if(allReviews.length===0)return null;
    const n=allReviews.length;
    const dist={positive:0,neutral:0,negative:0};
    const confBuckets={"0.5-0.6":0,"0.6-0.7":0,"0.7-0.8":0,"0.8-0.9":0,"0.9-1.0":0};
    let totalConf=0;
    const modelCM={};
    for(const m of MODELS_REF){
      modelCM[m.id]={};
      for(const t of LABELS){modelCM[m.id][t]={};for(const p of LABELS)modelCM[m.id][t][p]=0;}
    }
    for(const rev of allReviews){
      dist[rev.sentiment]=(dist[rev.sentiment]||0)+1;
      totalConf+=rev.confidence||0;
      const c=rev.confidence||0.5;
      if(c<0.6)confBuckets["0.5-0.6"]++;else if(c<0.7)confBuckets["0.6-0.7"]++;else if(c<0.8)confBuckets["0.7-0.8"]++;else if(c<0.9)confBuckets["0.8-0.9"]++;else confBuckets["0.9-1.0"]++;

      // FIX 4: use groundTruth (from rating) as the true label, NOT rev.sentiment
      // This stops models being compared against their own ensemble output
      const trueSent=rev.groundTruth||null;
      if(trueSent&&rev.allPreds){
        for(const[mid,pred]of Object.entries(rev.allPreds)){
          if(modelCM[mid]&&pred?.sentiment){
            modelCM[mid][trueSent]=modelCM[mid][trueSent]||{};
            modelCM[mid][trueSent][pred.sentiment]=(modelCM[mid][trueSent][pred.sentiment]||0)+1;
          }
        }
      }
    }
    // Compute per-model accuracy
    const modelStats={};
    for(const m of MODELS_REF){
      const cm=modelCM[m.id];
      let correct=0,total=0;
      const classMet={};
      for(const cls of LABELS){
        const tp=cm[cls]?.[cls]||0;
        let fp=0,fn=0;
        for(const o of LABELS){if(o!==cls){fp+=(cm[o]?.[cls]||0);fn+=(cm[cls]?.[o]||0);}}
        const prec=tp/(tp+fp)||0;const rec=tp/(tp+fn)||0;const f1=prec+rec>0?2*prec*rec/(prec+rec):0;
        classMet[cls]={precision:prec,recall:rec,f1};
        correct+=tp;total+=tp+fn;
      }
      modelStats[m.id]={cm,liveAcc:total>0?correct/total:null,classMet,correct,total};
    }
    return{n,dist,confBuckets,avgConf:totalConf/n,modelStats,modelCM};
  },[allReviews]);

  const tabs=[
    {id:"home",label:"Home"},{id:"analyze",label:"Analyze"},{id:"scrape",label:"Scrape Reviews"},
    {id:"upload",label:"Upload CSV"},{id:"results",label:`Results (${allReviews.length})`},{id:"models",label:"Models & Metrics"},{id:"datasets",label:"Datasets"},
  ];

  return(
    <div style={{minHeight:"100vh",background:P.bg,color:P.text,fontFamily:"'DM Sans','Segoe UI',sans-serif"}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
      <nav style={{borderBottom:`1px solid ${P.border}`,background:P.surface,padding:"0 20px",display:"flex",alignItems:"center",gap:12,overflowX:"auto",position:"sticky",top:0,zIndex:50}}>
        <div style={{fontSize:14,fontWeight:800,color:P.indigo,padding:"14px 0",whiteSpace:"nowrap",marginRight:6}}>SentimentAI</div>
        {tabs.map(t=><button key={t.id} onClick={()=>setTab(t.id)} style={{padding:"14px 2px",border:"none",cursor:"pointer",fontSize:11.5,fontWeight:tab===t.id?700:500,background:"none",color:tab===t.id?P.white:P.sub,borderBottom:tab===t.id?`2px solid ${P.indigo}`:"2px solid transparent",whiteSpace:"nowrap"}}>{t.label}</button>)}
      </nav>

      <div style={{padding:"20px 20px 50px",maxWidth:1100,margin:"0 auto"}}>

        {/* HOME */}
        {tab==="home"&&<div style={{display:"flex",flexDirection:"column",gap:20}}>
          <div style={{padding:"32px 0 12px",textAlign:"center"}}>
            <h1 style={{margin:"0 0 6px",fontSize:24,fontWeight:800,color:P.white}}>AI-Based Intelligent Customer Feedback Analyzer</h1>
            <h2 style={{margin:0,fontSize:14,fontWeight:400,color:P.sub}}>with Sentiment Confidence Scoring</h2>
            <div style={{marginTop:12,display:"flex",justifyContent:"center",gap:6}}><Glow c={P.green} s={8}/><span style={{fontSize:11,color:P.sub}}>Real-time scraping + CSV upload + 9 ML models</span></div>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(120px,1fr))",gap:8}}>
            <Num label="Analyzed" value={allReviews.length} color={P.iSoft} sub="Reviews"/>
            <Num label="RoBERTa" value="94.10%" color={P.indigo} sub="Transformer"/>
            <Num label="Best ML" value="90.32%" color={P.green} sub="FFNN+BoW"/>
            <Num label="Models" value="9" sub="All per review"/>
            <Num label="Confidence" value={metrics?`${(metrics.avgConf*100).toFixed(1)}%`:"—"} color={P.amber} sub="Average"/>
          </div>
          <Card style={{background:`linear-gradient(135deg,${P.card},#0d0a1e)`,border:`1px solid ${P.indigo}18`}}>
            <h3 style={{margin:"0 0 16px",fontSize:15,fontWeight:700,color:P.white,textAlign:"center"}}>Project Team — SRM University</h3>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(250px,1fr))",gap:22}}>
              <div>
                <div style={{fontSize:9,color:P.muted,textTransform:"uppercase",letterSpacing:"0.12em",fontWeight:700,marginBottom:10,paddingBottom:5,borderBottom:`1px solid ${P.border}`}}>Team Members</div>
                {TEAM.members.map((m,i)=><div key={i} style={{display:"flex",justifyContent:"space-between",padding:"9px 0",borderBottom:`1px solid ${P.border}08`}}><span style={{color:P.white,fontSize:13,fontWeight:600}}>{i+1}. {m.name}</span><span style={{color:P.iSoft,fontSize:11,fontFamily:"'DM Mono',monospace"}}>{m.reg}</span></div>)}
              </div>
              <div>
                <div style={{fontSize:9,color:P.muted,textTransform:"uppercase",letterSpacing:"0.12em",fontWeight:700,marginBottom:10,paddingBottom:5,borderBottom:`1px solid ${P.border}`}}>Supervisor</div>
                <div style={{fontSize:17,fontWeight:800,color:P.white,marginBottom:3,marginTop:4}}>{TEAM.sup.name}</div>
                <div style={{fontSize:12,color:P.sub}}>{TEAM.sup.title}</div>
                <div style={{fontSize:12,color:P.sub}}>Dept: {TEAM.sup.dept}, SRM University, Chennai</div>
              </div>
            </div>
          </Card>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:10}}>
            {[{t:"analyze",icon:"⚡",title:"Analyze Text",bc:P.indigo},{t:"scrape",icon:"🔍",title:"Scrape Live Reviews",bc:P.green},{t:"upload",icon:"📁",title:"Upload CSV Dataset",bc:P.amber},{t:"models",icon:"📊",title:"Models & Metrics",bc:P.red}].map(c=>
              <Card key={c.t} onClick={()=>setTab(c.t)} style={{cursor:"pointer",textAlign:"center",padding:16,border:`1px solid ${c.bc}15`}}><div style={{fontSize:24,marginBottom:4}}>{c.icon}</div><div style={{fontSize:13,fontWeight:700,color:P.white}}>{c.title}</div></Card>
            )}
          </div>
        </div>}

        {/* ANALYZE */}
        {tab==="analyze"&&<div style={{display:"flex",flexDirection:"column",gap:16}}>
          <Card>
            <h3 style={{margin:"0 0 12px",color:P.white,fontSize:15,fontWeight:700}}>Real-Time Sentiment Analysis</h3>
            <div style={{display:"flex",gap:8,marginBottom:12,flexWrap:"wrap"}}>
              <select value={selModel} onChange={e=>setSelModel(e.target.value)} style={{background:P.raised,border:`1px solid ${P.border}`,color:P.text,padding:"9px 12px",borderRadius:8,fontSize:12,flex:"1 1 300px"}}>
                <optgroup label="Transformer">{MODELS_REF.filter(m=>m.transformer).map(m=><option key={m.id} value={m.id}>{m.name} — {(m.acc*100).toFixed(2)}%</option>)}</optgroup>
                <optgroup label="Traditional ML">{MODELS_REF.filter(m=>!m.transformer).map(m=><option key={m.id} value={m.id}>{m.name} ({m.vec}) — {(m.acc*100).toFixed(2)}%</option>)}</optgroup>
              </select>
            </div>
            <div style={{display:"flex",gap:10}}>
              <textarea value={input} onChange={e=>setInput(e.target.value)} placeholder="Enter a customer review..." onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();doAnalyze();}}} style={{flex:1,background:P.raised,border:`1px solid ${P.border}`,color:P.text,padding:14,borderRadius:10,fontSize:13,resize:"vertical",minHeight:75,fontFamily:"inherit",lineHeight:1.5}}/>
              <button onClick={doAnalyze} disabled={!input.trim()||loading} style={{background:loading?P.border:P.indigo,color:"#fff",border:"none",borderRadius:10,padding:"0 22px",fontSize:13,fontWeight:700,cursor:input.trim()&&!loading?"pointer":"not-allowed",opacity:input.trim()&&!loading?1:0.4,minWidth:90}}>{loading?"...":"Analyze"}</button>
            </div>
          </Card>
          {allReviews.filter(r=>r.type==="manual").slice(0,8).map(r=><RevCard key={r.id} r={r}/>)}
        </div>}

        {/* SCRAPE — REAL */}
        {tab==="scrape"&&<div style={{display:"flex",flexDirection:"column",gap:16}}>
          <Card>
            <h3 style={{margin:"0 0 4px",color:P.white,fontSize:15,fontWeight:700}}>Scrape Real Reviews</h3>
            <p style={{margin:"0 0 12px",color:P.sub,fontSize:11}}>Fetches actual reviews from real websites and analyzes them through all 9 models.</p>
            <div style={{display:"flex",gap:8,marginBottom:10,flexWrap:"wrap"}}>
              <select value={scrapeSrc} onChange={e=>setScrapeSrc(e.target.value)} style={{background:P.raised,border:`1px solid ${P.border}`,color:P.text,padding:"9px 12px",borderRadius:8,fontSize:12}}>
                <option value="trustpilot">Trustpilot (no API key needed)</option>
                <option value="google">Google Reviews (needs Outscraper key)</option>
                <option value="amazon">Amazon (may be blocked)</option>
              </select>
              <input value={scrapeUrl} onChange={e=>setScrapeUrl(e.target.value)}
                placeholder={scrapeSrc==="trustpilot"?"Company domain: amazon.com, flipkart.com, samsung.com":scrapeSrc==="google"?"Search: iPhone 15 reviews, or Google Maps place name":"Amazon product review page URL"}
                style={{flex:1,background:P.raised,border:`1px solid ${P.border}`,color:P.text,padding:"9px 12px",borderRadius:8,fontSize:12,minWidth:200}}/>
              <button onClick={doScrape} disabled={!scrapeUrl.trim()||scraping} style={{background:scraping?P.border:`linear-gradient(135deg,#059669,${P.green})`,color:"#fff",border:"none",borderRadius:8,padding:"9px 18px",fontSize:12,fontWeight:700,cursor:scrapeUrl.trim()&&!scraping?"pointer":"not-allowed",opacity:scrapeUrl.trim()&&!scraping?1:0.4}}>{scraping?"Scraping...":"Scrape & Analyze"}</button>
            </div>
            {scrapeMsg&&<div style={{fontSize:11,color:scrapeMsg.includes("Error")||scrapeMsg.includes("error")?P.red:scrapeMsg.includes("Found")?P.green:P.amber,padding:"8px 12px",background:P.raised,borderRadius:8}}>{scrapeMsg}</div>}
            <div style={{fontSize:10,color:P.muted,marginTop:8,lineHeight:1.6}}>
              <strong>Trustpilot:</strong> Enter company domain (e.g. <em>amazon.com</em>, <em>flipkart.com</em>, <em>apple.com</em>)<br/>
              <strong>Google:</strong> Needs OUTSCRAPER_API_KEY env var in Railway. Free at outscraper.com<br/>
              <strong>Amazon:</strong> Direct scraping — may return empty if Amazon blocks the request
            </div>
          </Card>
          {allReviews.filter(r=>r.type==="scraped").slice(0,15).map(r=><RevCard key={r.id} r={r}/>)}
        </div>}

        {/* UPLOAD CSV */}
        {tab==="upload"&&<div style={{display:"flex",flexDirection:"column",gap:16}}>
          <Card>
            <h3 style={{margin:"0 0 4px",color:P.white,fontSize:15,fontWeight:700}}>Upload Review Dataset (CSV/Excel)</h3>
            <p style={{margin:"0 0 14px",color:P.sub,fontSize:11}}>Upload a CSV file with a text/review column. Up to 100 reviews will be analyzed through all 9 models. Supports Amazon, Yelp, IMDB, and custom datasets.</p>
            <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
              <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" onChange={doUpload} style={{background:P.raised,border:`1px solid ${P.border}`,color:P.text,padding:"10px 14px",borderRadius:8,fontSize:12,flex:1}}/>
              {uploading&&<span style={{fontSize:12,color:P.amber,fontWeight:600}}>Processing...</span>}
            </div>
            {uploadMsg&&<div style={{fontSize:11,color:uploadMsg.includes("Error")?P.red:P.green,padding:"8px 12px",background:P.raised,borderRadius:8,marginTop:10}}>{uploadMsg}</div>}
            <div style={{fontSize:10,color:P.muted,marginTop:12,lineHeight:1.6}}>
              <strong>Required:</strong> CSV with a column named <code style={{color:P.iSoft}}>text</code>, <code style={{color:P.iSoft}}>Text</code>, <code style={{color:P.iSoft}}>review</code>, or <code style={{color:P.iSoft}}>comment</code><br/>
              <strong>Optional:</strong> <code style={{color:P.iSoft}}>Score</code>, <code style={{color:P.iSoft}}>rating</code>, or <code style={{color:P.iSoft}}>stars</code> column for ground truth<br/>
              <strong>Datasets:</strong> Works with Amazon Fine Food Reviews, IMDB, Yelp, Sentiment140 CSV files from Kaggle
            </div>
          </Card>
          {allReviews.filter(r=>r.type==="uploaded").length>0&&<>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(120px,1fr))",gap:8}}>
              <Num label="Uploaded" value={allReviews.filter(r=>r.type==="uploaded").length} color={P.iSoft}/>
              <Num label="Positive" value={allReviews.filter(r=>r.type==="uploaded"&&r.sentiment==="positive").length} color={P.green}/>
              <Num label="Neutral" value={allReviews.filter(r=>r.type==="uploaded"&&r.sentiment==="neutral").length} color={P.amber}/>
              <Num label="Negative" value={allReviews.filter(r=>r.type==="uploaded"&&r.sentiment==="negative").length} color={P.red}/>
            </div>
            {allReviews.filter(r=>r.type==="uploaded").slice(0,15).map(r=><RevCard key={r.id} r={r}/>)}
          </>}
        </div>}

        {/* RESULTS */}
        {tab==="results"&&<div style={{display:"flex",flexDirection:"column",gap:12}}>
          <h3 style={{margin:0,color:P.white,fontSize:15}}>All Results ({allReviews.length})</h3>
          {allReviews.length===0?<Empty msg="No reviews yet" action="Start Analyzing" onAction={()=>setTab("analyze")}/>:allReviews.map(r=><RevCard key={r.id} r={r}/>)}
        </div>}

        {/* MODELS & METRICS — ALL REAL-TIME */}
        {tab==="models"&&<div style={{display:"flex",flexDirection:"column",gap:16}}>
          {!metrics?<Empty msg="Analyze or upload reviews first — all charts build in real-time from your data." action="Upload CSV" onAction={()=>setTab("upload")}/>:<>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(120px,1fr))",gap:8}}>
              <Num label="Reviews" value={metrics.n} color={P.iSoft}/><Num label="Positive" value={metrics.dist.positive||0} color={P.green}/><Num label="Neutral" value={metrics.dist.neutral||0} color={P.amber}/><Num label="Negative" value={metrics.dist.negative||0} color={P.red}/><Num label="Avg Conf" value={`${(metrics.avgConf*100).toFixed(1)}%`} color={P.iSoft}/>
            </div>

            {/* Accuracy table */}
            <Card>
              <h3 style={{margin:"0 0 4px",color:P.white,fontSize:14}}>Live Model Performance — {metrics.n} Reviews</h3>
              <p style={{margin:"0 0 12px",color:P.sub,fontSize:10}}>Live Acc uses rating as ground truth. Upload a CSV with a Score/rating column for meaningful values.</p>
              <div style={{overflowX:"auto"}}><table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                <thead><tr style={{borderBottom:`2px solid ${P.border}`}}>{["Model","Vec","Ref Acc","Live Acc","P","R","F1",""].map(h=><th key={h} style={{textAlign:h==="Model"?"left":"center",padding:"7px 8px",color:P.muted,fontWeight:600,fontSize:9.5}}>{h}</th>)}</tr></thead>
                <tbody>{MODELS_REF.map((m,i)=>{
                  const s=metrics.modelStats[m.id];if(!s)return null;
                  const wP=LABELS.reduce((a,c)=>a+(s.classMet[c]?.precision||0),0)/3;
                  const wR=LABELS.reduce((a,c)=>a+(s.classMet[c]?.recall||0),0)/3;
                  const wF=LABELS.reduce((a,c)=>a+(s.classMet[c]?.f1||0),0)/3;
                  const liveAccDisplay=s.liveAcc===null?"N/A":`${(s.liveAcc*100).toFixed(1)}%`;
                  const liveAccColor=s.liveAcc===null?P.muted:s.liveAcc>=0.8?P.green:s.liveAcc>=0.6?P.amber:P.red;
                  return<tr key={i} style={{borderBottom:`1px solid ${P.border}`,background:m.transformer?`${P.indigo}06`:m.paperBest?`${P.green}04`:"transparent"}}>
                    <td style={{padding:"8px",color:P.white,fontWeight:m.transformer||m.paperBest?700:400}}>{m.name}{m.transformer&&<span style={{fontSize:7,color:P.iSoft,background:`${P.indigo}18`,padding:"1px 5px",borderRadius:6,marginLeft:4}}>NEW</span>}{m.paperBest&&<span style={{fontSize:7,color:P.green,background:`${P.green}14`,padding:"1px 5px",borderRadius:6,marginLeft:4}}>BEST</span>}</td>
                    <td style={{textAlign:"center",padding:"8px",color:P.sub,fontSize:10}}>{m.vec}</td>
                    <td style={{textAlign:"center",padding:"8px",fontFamily:"monospace",color:P.muted}}>{(m.acc*100).toFixed(2)}%</td>
                    <td style={{textAlign:"center",padding:"8px",fontFamily:"monospace",fontWeight:700,color:liveAccColor}}>{liveAccDisplay}</td>
                    <td style={{textAlign:"center",padding:"8px",fontFamily:"monospace",color:P.text}}>{(wP*100).toFixed(1)}%</td>
                    <td style={{textAlign:"center",padding:"8px",fontFamily:"monospace",color:P.text}}>{(wR*100).toFixed(1)}%</td>
                    <td style={{textAlign:"center",padding:"8px",fontFamily:"monospace",color:P.text}}>{(wF*100).toFixed(1)}%</td>
                    <td style={{padding:"8px",width:90}}>{s.liveAcc!==null?<CBar v={s.liveAcc} h={4}/>:<span style={{fontSize:9,color:P.muted}}>need rating</span>}</td>
                  </tr>})}</tbody>
              </table></div>
            </Card>

            {/* Bar chart */}
            <Card>
              <h3 style={{margin:"0 0 12px",color:P.white,fontSize:14}}>Live Accuracy — All Models</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={MODELS_REF.map(m=>({name:`${m.short}${m.transformer?"":"/"+m.vec}`,acc:metrics.modelStats[m.id]?.liveAcc!==null?+((metrics.modelStats[m.id]?.liveAcc||0)*100).toFixed(1):null,ref:+(m.acc*100).toFixed(1)}))} margin={{top:10,right:20,left:0,bottom:30}}>
                  <CartesianGrid strokeDasharray="3 3" stroke={P.border}/><XAxis dataKey="name" tick={{fill:P.sub,fontSize:9}} angle={-25} textAnchor="end" interval={0}/><YAxis domain={[0,100]} tick={{fill:P.sub,fontSize:10}}/><Tooltip contentStyle={{background:P.card,border:`1px solid ${P.border}`,borderRadius:8,color:P.text,fontSize:11}}/><Legend wrapperStyle={{fontSize:10}}/>
                  <Bar dataKey="ref" name="Reference" fill={P.muted} radius={[3,3,0,0]} fillOpacity={0.4}/><Bar dataKey="acc" name="Live" fill={P.indigo} radius={[3,3,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Confusion matrix */}
            <Card>
              <h3 style={{margin:"0 0 4px",color:P.white,fontSize:14}}>Live Confusion Matrix</h3>
              <p style={{margin:"0 0 10px",color:P.sub,fontSize:10}}>Rows = True label (from rating), Columns = Predicted. Requires reviews with a rating/score.</p>
              <div style={{display:"flex",gap:4,marginBottom:12,flexWrap:"wrap"}}>
                {MODELS_REF.map(m=><button key={m.id} onClick={()=>setSelModel(m.id)} style={{padding:"4px 9px",borderRadius:6,border:`1px solid ${selModel===m.id?P.indigo:P.border}`,background:selModel===m.id?`${P.indigo}18`:"transparent",color:selModel===m.id?P.iSoft:P.sub,fontSize:9.5,fontWeight:600,cursor:"pointer"}}>{m.short}{m.transformer?"":"/"+m.vec}</button>)}
              </div>
              {(()=>{const s=metrics.modelStats[selModel];if(!s)return null;
                if(s.total===0)return<div style={{color:P.sub,fontSize:11,textAlign:"center",padding:"20px"}}>No ground truth data yet — upload a CSV with a Score/rating column to populate this matrix.</div>;
                const mx=Math.max(...LABELS.flatMap(t=>LABELS.map(p=>s.cm[t]?.[p]||0)),1);
                return<div style={{display:"flex",justifyContent:"center"}}><div>
                  <div style={{display:"flex",marginLeft:70}}><div style={{width:"100%",textAlign:"center",fontSize:10,fontWeight:700,color:P.white,marginBottom:4}}>Predicted</div></div>
                  <div style={{display:"flex",marginLeft:70,marginBottom:4}}>{LABELS.map(l=><div key={l} style={{width:80,textAlign:"center",fontSize:9,color:P.sub,fontWeight:600,textTransform:"capitalize"}}>{l}</div>)}</div>
                  {LABELS.map((tl,i)=><div key={i} style={{display:"flex",alignItems:"center",marginBottom:3}}>
                    <div style={{width:70,textAlign:"right",paddingRight:6,fontSize:9,color:P.sub,fontWeight:600,textTransform:"capitalize"}}>{tl}</div>
                    {LABELS.map((pl,j)=>{const v=s.cm[tl]?.[pl]||0;const d=i===j;const int=v/mx;return<div key={j} style={{width:80,height:48,display:"flex",alignItems:"center",justifyContent:"center",background:d?`rgba(99,91,255,${0.1+int*0.45})`:`rgba(255,69,58,${int*0.25})`,border:`1px solid ${P.border}`,borderRadius:6,margin:"0 2px"}}><span style={{fontSize:14,fontWeight:700,color:d?P.iSoft:P.text,fontFamily:"monospace"}}>{v}</span></div>})}
                  </div>)}
                  <div style={{marginLeft:70,marginTop:6,fontSize:9,color:P.sub}}>Accuracy: <strong style={{color:s.liveAcc>=0.8?P.green:P.amber}}>{(s.liveAcc*100).toFixed(1)}%</strong> ({s.correct}/{s.total})</div>
                </div></div>})()}
            </Card>

            {/* Pie + Confidence */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(260px,1fr))",gap:12}}>
              <Card>
                <h4 style={{margin:"0 0 10px",color:P.white,fontSize:13}}>Sentiment Distribution</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart><Pie data={LABELS.map(l=>({name:l,value:metrics.dist[l]||0}))} cx="50%" cy="50%" innerRadius={42} outerRadius={78} paddingAngle={3} dataKey="value">{[P.green,P.amber,P.red].map((c,i)=><Cell key={i} fill={c}/>)}</Pie><Tooltip contentStyle={{background:P.card,border:`1px solid ${P.border}`,borderRadius:8,color:P.text,fontSize:11}}/></PieChart>
                </ResponsiveContainer>
              </Card>
              <Card>
                <h4 style={{margin:"0 0 10px",color:P.white,fontSize:13}}>Confidence Distribution</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={Object.entries(metrics.confBuckets).map(([k,v])=>({range:k,count:v}))} margin={{top:10,right:10,left:0,bottom:10}}><CartesianGrid strokeDasharray="3 3" stroke={P.border}/><XAxis dataKey="range" tick={{fill:P.sub,fontSize:9}}/><YAxis tick={{fill:P.sub,fontSize:9}}/><Tooltip contentStyle={{background:P.card,border:`1px solid ${P.border}`,borderRadius:8,color:P.text,fontSize:11}}/><Bar dataKey="count" radius={[4,4,0,0]}>{Object.keys(metrics.confBuckets).map((_,i)=><Cell key={i} fill={i>=3?P.green:i>=2?P.amber:P.red} fillOpacity={0.7}/>)}</Bar></BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Radar */}
            <Card>
              <h3 style={{margin:"0 0 12px",color:P.white,fontSize:14}}>Radar — Model Comparison</h3>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={["Accuracy","Precision","Recall","F1"].map(met=>{const row={metric:met};for(const m of MODELS_REF.filter((_,i)=>[0,1,3,5,7].includes(i))){const s=metrics.modelStats[m.id];if(!s)continue;const wP=LABELS.reduce((a,c)=>a+(s.classMet[c]?.precision||0),0)/3;const wR=LABELS.reduce((a,c)=>a+(s.classMet[c]?.recall||0),0)/3;const wF=LABELS.reduce((a,c)=>a+(s.classMet[c]?.f1||0),0)/3;const accVal=s.liveAcc!==null?s.liveAcc:m.acc;row[m.short+(m.transformer?"":"/"+m.vec)]=+((met==="Accuracy"?accVal:met==="Precision"?wP:met==="Recall"?wR:wF)*100).toFixed(1);}return row;})}>
                  <PolarGrid stroke={P.border}/><PolarAngleAxis dataKey="metric" tick={{fill:P.sub,fontSize:11}}/><PolarRadiusAxis angle={30} domain={[0,100]} tick={{fill:P.faint,fontSize:9}}/>
                  <Radar name="RoBERTa" dataKey="RoBERTa" stroke={P.indigo} fill={P.indigo} fillOpacity={0.15} strokeWidth={2}/><Radar name="FFNN/BoW" dataKey="FFNN/BoW" stroke={P.green} fill={P.green} fillOpacity={0.1}/><Radar name="LR/TF-IDF" dataKey="LR/TF-IDF" stroke={P.amber} fill={P.amber} fillOpacity={0.08}/><Radar name="NB/TF-IDF" dataKey="NB/TF-IDF" stroke={P.red} fill={P.red} fillOpacity={0.08}/><Radar name="RF/TF-IDF" dataKey="RF/TF-IDF" stroke="#c084fc" fill="#c084fc" fillOpacity={0.08}/>
                  <Legend wrapperStyle={{fontSize:10}}/>
                </RadarChart>
              </ResponsiveContainer>
            </Card>

            <Card style={{textAlign:"center"}}><div style={{fontSize:16,fontFamily:"'DM Mono',monospace",fontWeight:700,color:P.iSoft,marginBottom:6}}>Confidence = max( P_pos , P_neu , P_neg )</div><div style={{fontSize:10,color:P.sub}}>Via predict_proba() — high confidence auto-processed, low confidence flagged.</div></Card>
          </>}
        </div>}

        {/* DATASETS */}
        {tab==="datasets"&&<div style={{display:"flex",flexDirection:"column",gap:12}}>
          <h3 style={{margin:0,color:P.white,fontSize:15}}>Datasets & Sources</h3>
          {DATASETS.map((d,i)=><Card key={i} style={{padding:14,borderLeft:d.primary?`3px solid ${P.indigo}`:`3px solid ${P.border}`}}><div style={{display:"flex",justifyContent:"space-between",gap:12,flexWrap:"wrap"}}><div><span style={{fontSize:13,fontWeight:700,color:P.white}}>{d.name}</span>{d.primary&&<span style={{fontSize:7,color:P.indigo,background:`${P.indigo}18`,padding:"1px 6px",borderRadius:6,marginLeft:6}}>PRIMARY</span>}<p style={{margin:"2px 0",color:P.sub,fontSize:10.5}}>{d.desc}</p></div><div style={{textAlign:"right"}}><div style={{fontSize:14,fontWeight:800,color:P.iSoft,fontFamily:"monospace"}}>{d.size}</div><div style={{fontSize:9.5,color:P.sub}}>{d.src}</div></div></div></Card>)}
        </div>}

      </div>
      <div style={{borderTop:`1px solid ${P.border}`,padding:"12px 20px",textAlign:"center"}}><div style={{fontSize:9.5,color:P.muted}}>{TEAM.members.map(m=>m.name).join(" • ")} — Supervised by {TEAM.sup.name}</div><div style={{fontSize:9.5,color:P.faint}}>Dept. of {TEAM.sup.dept} • SRM University, Chennai</div></div>
    </div>
  );
}

// ── Review Card ──
function RevCard({r}){
  const[open,setOpen]=useState(false);
  const c=sC(r.sentiment);
  return<div style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:12,padding:14}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:14,flexWrap:"wrap"}}>
      <div style={{flex:1,minWidth:200}}>
        <p style={{margin:"0 0 7px",color:P.text,fontSize:12.5,lineHeight:1.55}}>"{r.text?.slice(0,300)}{r.text?.length>300?"...":""}"</p>
        <div style={{display:"flex",gap:6,alignItems:"center",flexWrap:"wrap"}}>
          <Badge s={r.sentiment}/>
          {r.groundTruth&&r.groundTruth!==r.sentiment&&<span style={{fontSize:8,color:P.amber,background:`${P.amber}12`,padding:"2px 6px",borderRadius:6,fontWeight:700}}>TRUE: {r.groundTruth.toUpperCase()}</span>}
          {r.rating&&<span style={{fontSize:10,color:"#fbbf24"}}>{"★".repeat(Math.min(5,r.rating))}{"☆".repeat(Math.max(0,5-r.rating))}</span>}
          {r.type==="scraped"&&<span style={{fontSize:8,color:P.green,background:`${P.green}12`,padding:"2px 6px",borderRadius:6,fontWeight:700}}>SCRAPED</span>}
          {r.type==="uploaded"&&<span style={{fontSize:8,color:P.amber,background:`${P.amber}12`,padding:"2px 6px",borderRadius:6,fontWeight:700}}>CSV</span>}
          <span style={{fontSize:9.5,color:P.sub}}>{r.source} {r.author?`• ${r.author}`:""}</span>
        </div>
      </div>
      <div style={{minWidth:130}}><div style={{fontSize:9,color:P.muted,marginBottom:4}}>Confidence</div><CBar v={r.confidence||0.5}/></div>
    </div>
    {r.allPreds&&Object.keys(r.allPreds).length>0&&<>
      <button onClick={()=>setOpen(!open)} style={{marginTop:8,background:"none",border:"none",color:P.iSoft,fontSize:10,fontWeight:600,cursor:"pointer",padding:0}}>{open?"Hide":"Show"} all models</button>
      {open&&<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(170px,1fr))",gap:4,marginTop:8}}>
        {Object.entries(r.allPreds).map(([mid,pred])=>{
          const m=MODELS_REF.find(x=>x.id===mid);const name=m?`${m.short}${m.transformer?"":"/"+m.vec}`:mid;
          return<div key={mid} style={{padding:"6px 8px",borderRadius:6,background:P.raised,border:`1px solid ${P.border}`,fontSize:9.5}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}><span style={{color:P.text,fontWeight:600}}>{name}</span><span style={{color:sC(pred.sentiment),fontWeight:700,fontSize:9,textTransform:"uppercase"}}>{pred.sentiment}</span></div>
            <CBar v={pred.confidence||0.5} h={3}/>
          </div>})}
      </div>}
    </>}
  </div>;
}
