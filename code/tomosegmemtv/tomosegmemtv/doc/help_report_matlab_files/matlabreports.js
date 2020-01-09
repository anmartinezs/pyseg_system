/*matlabreports.js
 Copyright 2008-2009 The MathWorks, Inc.
 */

function toggleexpander(blockid) {
   block = document.getElementById(blockid);
   showMore = document.getElementById(blockid+"_button")
   arrow = document.getElementById(blockid+"_arrow");
   if (block.style.display === undefined || block.style.display == "undefined" || block.style.display == "")
    {
     block.style.display = "none";
    }
 
    if (block.style.display == "none")
    {
     // currently collapsed, so expand it.

     block.style.display = "block";
     showMore.innerHTML="show less";
     arrow.src="file://"+arrow_open; 
    }
    else 
    {
     // currently expanded, so collapse it.

     block.style.display = "none";
     showMore.innerHTML="show more";
     arrow.src="file://"+arrow_closed;
    }
}

function setRegexpPref(val) {
    var command = "internal.matlab.codetools.reports.setDoFixRegexp('" + val + "')";
    runMatlab(command);
}

function runMatlab(command) {
    document.location = "matlab:" + command;
}

function runreport(reportcommand) {
    document.location="matlab:" + reportcommand;
}

