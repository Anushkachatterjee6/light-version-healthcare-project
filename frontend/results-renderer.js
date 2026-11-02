// Shared result components used by the dashboard and Image Analysis Studio.
(() => {
  'use strict';
  function node(tag, text, cls) { const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(cls)el.className=cls; return el; }
  function fmt(v){ return typeof v==='number' ? (Number.isInteger(v)?String(v):v.toLocaleString(undefined,{maximumFractionDigits:4})) : String(v ?? '—'); }
  function cnnImage(source,caption,alt){
    const fig=node('figure',undefined,'cnn-image-card');fig.append(node('figcaption',caption));
    if(typeof source!=='string'||!/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=\s]+$/.test(source)){
      fig.append(node('p','Image unavailable for this result.','warning'));return fig;
    }
    const img=node('img');img.alt=alt;
    img.addEventListener('error',()=>{img.hidden=true;fig.append(node('p','Image could not be decoded. Run the CNN analysis again.','warning'));});
    img.src=source;fig.append(img);return fig;
  }
  function renderCNNImages(root,result){
    const section=node('section',undefined,'cnn-visuals');section.append(node('h3','CNN Image Results'));
    section.append(node('p','Classification CNN: each prediction labels the model input; this model does not generate a segmentation mask. Scores are model outputs, not calibrated clinical confidence.','schema'));
    const previews=Array.isArray(result.previews)?result.previews:[];
    for(const preview of previews){
      const sample=node('article',undefined,'cnn-sample');sample.append(node('h4',`${preview.case} · Slice ${preview.slice}`));
      const comparison=node('div',undefined,'cnn-comparison');
      comparison.append(cnnImage(preview.input_image,'Input Image (preprocessed model input)',`Input MRI for ${preview.case}, slice ${preview.slice}`));
      comparison.append(cnnImage(preview.ground_truth_mask,'Ground Truth'+(preview.actual_class?` · ${preview.actual_class}`:''),'Reference segmentation mask'));
      const prediction=cnnImage(preview.input_image,'CNN Prediction'+(preview.predicted_class?` · ${preview.predicted_class}`:''),'CNN classification shown on the same input image');
      const score=preview.predicted_class_score;
      prediction.append(node('p',Number.isFinite(score)?`Predicted-class score: ${(score*100).toFixed(2)}%`:'Prediction score unavailable.','cnn-score'));
      comparison.append(prediction);sample.append(comparison);section.append(sample);
    }
    if(!previews.length){
      // Older current-generation responses may have just the composite plot.
      const composite=(result.plots||[]).find(plot=>/input.*mask.*prediction/i.test(plot.title||''));
      if(composite)section.append(cnnImage('data:image/png;base64,'+composite.png,'Input Image / Ground Truth / CNN Prediction',composite.title));
      else section.append(node('p','No CNN images were returned. Restart the project backend to load the updated CNN endpoint, then run the CNN button again.','warning'));
    }
    root.append(section);
  }
  function renderTextAnnotations(root,annotations){
    if(!annotations||typeof annotations.text!=='string')return;
    const section=node('section',undefined,'text-annotations');section.append(node('h3','Clinical Text — Detected Mentions'));
    const text=annotations.text,paragraph=node('p',undefined,'annotated-source');let position=0;
    const spans=(annotations.spans||[]).slice().sort((a,b)=>a.start-b.start||b.end-a.end);
    for(const span of spans){
      if(!Number.isInteger(span.start)||!Number.isInteger(span.end)||span.start<position||span.end>text.length||span.end<=span.start)continue;
      paragraph.append(node('span',text.slice(position,span.start)));
      const mark=node('mark',text.slice(span.start,span.end),span.possible_negation?'mention-negated':'mention');
      mark.title=span.category+(span.possible_negation?' · possible negation cue':'')+' · detected mention, not a verified fact';
      paragraph.append(mark);position=span.end;
    }
    paragraph.append(node('span',text.slice(position)));section.append(paragraph);
    section.append(node('p',spans.length?'Highlighted mentions use the returned character offsets. Amber marks indicate a possible negation cue; inspect the entity table for category and method.':'No supported mentions were detected; text is shown unchanged.','schema'));
    root.append(section);
  }
  function render(root,result,cnn=false){
    root.replaceChildren();
    root.append(node('p',result.dataset,'dataset-notice'));
    if(cnn)renderCNNImages(root,result);
    renderTextAnnotations(root,result.annotations);
    const clusterRows=result.tables?.clusters||result.tables?.cluster_representatives||[];
    if(result.tutorial===10&&clusterRows.length){
      const section=node('section',undefined,'cluster-terms');section.append(node('h3','Cluster Representatives'));
      for(const cluster of clusterRows){const card=node('article');card.append(node('h4',`Cluster ${cluster.cluster} · ${cluster.size} training documents`),node('p',cluster.terms));section.append(card);}root.append(section);
    }
    const metrics=node('div',undefined,'metrics');
    for(const [key,value] of Object.entries(result.metrics||{})){const card=node('div',undefined,'metric');card.append(node('small',key.replaceAll('_',' ')),node('strong',fmt(value)));metrics.append(card);}root.append(metrics);
    for(const text of result.interpretation||[]) root.append(node('p',text,'interpretation'));
    for(const text of result.warnings||[]) root.append(node('p',text,'warning'));
    for(const text of result.visualization_notes||[]) root.append(node('p',text,'visualization-note'));
    if(cnn&&!(result.plots||[]).length)root.append(node('p','Training and evaluation plots are unavailable in this response. Restart the backend and rerun CNN.','warning'));
    if(cnn&&(result.plots||[]).length)root.append(node('h3','Training History & Evaluation'));
    const plots=node('div',undefined,'plots');
    for(const plot of result.plots||[]){
      const fig=node('figure',undefined,plot.wide?'plot-wide':undefined);fig.append(node('figcaption',plot.title));
      if(typeof plot.png!=='string'||!plot.png.trim()){fig.append(node('p','Visualization unavailable in this response.','warning'));}
      else {const img=node('img');img.alt=plot.title;
        img.addEventListener('error',()=>{img.hidden=true;fig.append(node('p','This visualization could not be decoded. Rerun the analysis.','warning'));});
        img.src='data:image/png;base64,'+plot.png;fig.append(img);}
      if(plot.caption)fig.append(node('p',plot.caption,'plot-caption'));
      plots.append(fig);
    }
    if(!plots.children.length&&!cnn&&!result.annotations)root.append(node('p','No plot applies to the returned results. Inspect the tables and any dataset warnings.','visualization-note'));
    root.append(plots);
    for(const [name,rows] of Object.entries(result.tables||{})){
      const details=node('details'); details.append(node('summary',`${name.replaceAll('_',' ')} · ${rows.length} rows`));
      const wrap=node('div',undefined,'table-wrap');
      if(!rows.length) wrap.append(node('p','No matching records.'));
      else {const table=node('table');const keys=[...new Set(rows.flatMap(Object.keys))];const head=node('tr');for(const key of keys)head.append(node('th',key));const thead=node('thead');thead.append(head);table.append(thead);const tbody=node('tbody');for(const row of rows){const tr=node('tr');for(const key of keys)tr.append(node('td',fmt(row[key])));tbody.append(tr);}table.append(tbody);wrap.append(table);}
      details.append(wrap);root.append(details);
    }
  }
  globalThis.HERResults = Object.freeze({render});
})();
