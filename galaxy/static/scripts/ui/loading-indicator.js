!function(a,b){"function"==typeof define&&define.amd?define([],b):a.LoadingIndicator=b()}(this,function(){function a(a,b){function c(){var c=['<div class="loading-indicator">','<div class="loading-indicator-text">','<span class="fa fa-spinner fa-spin fa-lg"></span>','<span class="loading-indicator-message">loading...</span>',"</div>","</div>"].join("\n"),d=$(c).hide().css(b.css||{position:"fixed"}),e=d.children(".loading-indicator-text");return b.cover?(d.css({"z-index":2,top:a.css("top"),bottom:a.css("bottom"),left:a.css("left"),right:a.css("right"),opacity:.5,"background-color":"white","text-align":"center"}),e=d.children(".loading-indicator-text").css({"margin-top":"20px"})):(e=d.children(".loading-indicator-text").css({margin:"12px 0px 0px 10px",opacity:"0.85",color:"grey"}),e.children(".loading-indicator-message").css({margin:"0px 8px 0px 0px","font-style":"italic"})),d}var d=this;return b=jQuery.extend({cover:!1},b||{}),d.show=function(b,e,f){return b=b||"loading...",e=e||"fast",a.parent().find(".loading-indicator").remove(),d.$indicator=c().insertBefore(a),d.message(b),d.$indicator.fadeIn(e,f),d},d.message=function(a){d.$indicator.find("i").text(a)},d.hide=function(a,b){return a=a||"fast",d.$indicator&&d.$indicator.length?d.$indicator.fadeOut(a,function(){d.$indicator.remove(),b&&b()}):b&&b(),d},d}return a});
//# sourceMappingURL=../../maps/ui/loading-indicator.js.map