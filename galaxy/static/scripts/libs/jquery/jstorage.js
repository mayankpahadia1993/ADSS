!function(){function a(){var a=!1;if("localStorage"in window)try{window.localStorage.setItem("_tmptest","tmpval"),a=!0,window.localStorage.removeItem("_tmptest")}catch(b){}if(a)try{window.localStorage&&(v=window.localStorage,y="localStorage",B=v.jStorage_update)}catch(e){}else if("globalStorage"in window)try{window.globalStorage&&(v="localhost"==window.location.hostname?window.globalStorage["localhost.localdomain"]:window.globalStorage[window.location.hostname],y="globalStorage",B=v.jStorage_update)}catch(f){}else{if(w=document.createElement("link"),!w.addBehavior)return void(w=null);w.style.behavior="url(#default#userData)",document.getElementsByTagName("head")[0].appendChild(w);try{w.load("jStorage")}catch(g){w.setAttribute("jStorage","{}"),w.save("jStorage"),w.load("jStorage")}var i="{}";try{i=w.getAttribute("jStorage")}catch(j){}try{B=w.getAttribute("jStorage_update")}catch(m){}v.jStorage=i,y="userDataBehavior"}h(),k(),c(),l(),"addEventListener"in window&&window.addEventListener("pageshow",function(a){a.persisted&&d()},!1)}function b(){var a="{}";if("userDataBehavior"==y){w.load("jStorage");try{a=w.getAttribute("jStorage")}catch(b){}try{B=w.getAttribute("jStorage_update")}catch(c){}v.jStorage=a}h(),k(),l()}function c(){"localStorage"==y||"globalStorage"==y?"addEventListener"in window?window.addEventListener("storage",d,!1):document.attachEvent("onstorage",d):"userDataBehavior"==y&&setInterval(d,1e3)}function d(){var a;clearTimeout(A),A=setTimeout(function(){if("localStorage"==y||"globalStorage"==y)a=v.jStorage_update;else if("userDataBehavior"==y){w.load("jStorage");try{a=w.getAttribute("jStorage_update")}catch(b){}}a&&a!=B&&(B=a,e())},25)}function e(){var a,c=s.parse(s.stringify(u.__jstorage_meta.CRC32));b(),a=s.parse(s.stringify(u.__jstorage_meta.CRC32));var d,e=[],g=[];for(d in c)if(c.hasOwnProperty(d)){if(!a[d]){g.push(d);continue}c[d]!=a[d]&&"2."==String(c[d]).substr(0,2)&&e.push(d)}for(d in a)a.hasOwnProperty(d)&&(c[d]||e.push(d));f(e,"updated"),f(g,"deleted")}function f(a,b){if(a=[].concat(a||[]),"flushed"==b){a=[];for(var c in z)z.hasOwnProperty(c)&&a.push(c);b="deleted"}for(var d=0,e=a.length;e>d;d++){if(z[a[d]])for(var f=0,g=z[a[d]].length;g>f;f++)z[a[d]][f](a[d],b);if(z["*"])for(var f=0,g=z["*"].length;g>f;f++)z["*"][f](a[d],b)}}function g(){var a=(+new Date).toString();if("localStorage"==y||"globalStorage"==y)try{v.jStorage_update=a}catch(b){y=!1}else"userDataBehavior"==y&&(w.setAttribute("jStorage_update",a),w.save("jStorage"));d()}function h(){if(v.jStorage)try{u=s.parse(String(v.jStorage))}catch(a){v.jStorage="{}"}else v.jStorage="{}";x=v.jStorage?String(v.jStorage).length:0,u.__jstorage_meta||(u.__jstorage_meta={}),u.__jstorage_meta.CRC32||(u.__jstorage_meta.CRC32={})}function i(){n();try{v.jStorage=s.stringify(u),w&&(w.setAttribute("jStorage",v.jStorage),w.save("jStorage")),x=v.jStorage?String(v.jStorage).length:0}catch(a){}}function j(a){if(!a||"string"!=typeof a&&"number"!=typeof a)throw new TypeError("Key name must be string or numeric");if("__jstorage_meta"==a)throw new TypeError("Reserved key name");return!0}function k(){var a,b,c,d,e=1/0,h=!1,j=[];if(clearTimeout(t),u.__jstorage_meta&&"object"==typeof u.__jstorage_meta.TTL){a=+new Date,c=u.__jstorage_meta.TTL,d=u.__jstorage_meta.CRC32;for(b in c)c.hasOwnProperty(b)&&(c[b]<=a?(delete c[b],delete d[b],delete u[b],h=!0,j.push(b)):c[b]<e&&(e=c[b]));e!=1/0&&(t=setTimeout(k,e-a)),h&&(i(),g(),f(j,"deleted"))}}function l(){var a,b;if(u.__jstorage_meta.PubSub){var c,d=D;for(a=b=u.__jstorage_meta.PubSub.length-1;a>=0;a--)c=u.__jstorage_meta.PubSub[a],c[0]>D&&(d=c[0],m(c[1],c[2]));D=d}}function m(a,b){if(C[a])for(var c=0,d=C[a].length;d>c;c++)C[a][c](a,s.parse(s.stringify(b)))}function n(){if(u.__jstorage_meta.PubSub){for(var a=+new Date-2e3,b=0,c=u.__jstorage_meta.PubSub.length;c>b;b++)if(u.__jstorage_meta.PubSub[b][0]<=a){u.__jstorage_meta.PubSub.splice(b,u.__jstorage_meta.PubSub.length-b);break}u.__jstorage_meta.PubSub.length||delete u.__jstorage_meta.PubSub}}function o(a,b){u.__jstorage_meta||(u.__jstorage_meta={}),u.__jstorage_meta.PubSub||(u.__jstorage_meta.PubSub=[]),u.__jstorage_meta.PubSub.unshift([+new Date,a,b]),i(),g()}function p(a,b){for(var c,d=a.length,e=b^d,f=0;d>=4;)c=255&a.charCodeAt(f)|(255&a.charCodeAt(++f))<<8|(255&a.charCodeAt(++f))<<16|(255&a.charCodeAt(++f))<<24,c=1540483477*(65535&c)+((1540483477*(c>>>16)&65535)<<16),c^=c>>>24,c=1540483477*(65535&c)+((1540483477*(c>>>16)&65535)<<16),e=1540483477*(65535&e)+((1540483477*(e>>>16)&65535)<<16)^c,d-=4,++f;switch(d){case 3:e^=(255&a.charCodeAt(f+2))<<16;case 2:e^=(255&a.charCodeAt(f+1))<<8;case 1:e^=255&a.charCodeAt(f),e=1540483477*(65535&e)+((1540483477*(e>>>16)&65535)<<16)}return e^=e>>>13,e=1540483477*(65535&e)+((1540483477*(e>>>16)&65535)<<16),e^=e>>>15,e>>>0}var q="0.4.4",r=window.jQuery||window.$||(window.$={}),s={parse:window.JSON&&(window.JSON.parse||window.JSON.decode)||String.prototype.evalJSON&&function(a){return String(a).evalJSON()}||r.parseJSON||r.evalJSON,stringify:Object.toJSON||window.JSON&&(window.JSON.stringify||window.JSON.encode)||r.toJSON};if(!("parse"in s&&"stringify"in s))throw new Error("No JSON support found, include //cdnjs.cloudflare.com/ajax/libs/json2/20110223/json2.js to page");var t,u={__jstorage_meta:{CRC32:{}}},v={jStorage:"{}"},w=null,x=0,y=!1,z={},A=!1,B=0,C={},D=+new Date,E={isXML:function(a){var b=(a?a.ownerDocument||a:0).documentElement;return b?"HTML"!==b.nodeName:!1},encode:function(a){if(!this.isXML(a))return!1;try{return(new XMLSerializer).serializeToString(a)}catch(b){try{return a.xml}catch(c){}}return!1},decode:function(a){var b,c="DOMParser"in window&&(new DOMParser).parseFromString||window.ActiveXObject&&function(a){var b=new ActiveXObject("Microsoft.XMLDOM");return b.async="false",b.loadXML(a),b};return c?(b=c.call("DOMParser"in window&&new DOMParser||window,a,"text/xml"),this.isXML(b)?b:!1):!1}};r.jStorage={version:q,set:function(a,b,c){if(j(a),c=c||{},"undefined"==typeof b)return this.deleteKey(a),b;if(E.isXML(b))b={_is_xml:!0,xml:E.encode(b)};else{if("function"==typeof b)return void 0;b&&"object"==typeof b&&(b=s.parse(s.stringify(b)))}return u[a]=b,u.__jstorage_meta.CRC32[a]="2."+p(s.stringify(b),2538058380),this.setTTL(a,c.TTL||0),f(a,"updated"),b},get:function(a,b){return j(a),a in u?u[a]&&"object"==typeof u[a]&&u[a]._is_xml?E.decode(u[a].xml):u[a]:"undefined"==typeof b?null:b},deleteKey:function(a){return j(a),a in u?(delete u[a],"object"==typeof u.__jstorage_meta.TTL&&a in u.__jstorage_meta.TTL&&delete u.__jstorage_meta.TTL[a],delete u.__jstorage_meta.CRC32[a],i(),g(),f(a,"deleted"),!0):!1},setTTL:function(a,b){var c=+new Date;return j(a),b=Number(b)||0,a in u?(u.__jstorage_meta.TTL||(u.__jstorage_meta.TTL={}),b>0?u.__jstorage_meta.TTL[a]=c+b:delete u.__jstorage_meta.TTL[a],i(),k(),g(),!0):!1},getTTL:function(a){var b,c=+new Date;return j(a),a in u&&u.__jstorage_meta.TTL&&u.__jstorage_meta.TTL[a]?(b=u.__jstorage_meta.TTL[a]-c,b||0):0},flush:function(){return u={__jstorage_meta:{CRC32:{}}},i(),g(),f(null,"flushed"),!0},storageObj:function(){function a(){}return a.prototype=u,new a},index:function(){var a,b=[];for(a in u)u.hasOwnProperty(a)&&"__jstorage_meta"!=a&&b.push(a);return b},storageSize:function(){return x},currentBackend:function(){return y},storageAvailable:function(){return!!y},listenKeyChange:function(a,b){j(a),z[a]||(z[a]=[]),z[a].push(b)},stopListening:function(a,b){if(j(a),z[a]){if(!b)return void delete z[a];for(var c=z[a].length-1;c>=0;c--)z[a][c]==b&&z[a].splice(c,1)}},subscribe:function(a,b){if(a=(a||"").toString(),!a)throw new TypeError("Channel not defined");C[a]||(C[a]=[]),C[a].push(b)},publish:function(a,b){if(a=(a||"").toString(),!a)throw new TypeError("Channel not defined");o(a,b)},reInit:function(){b()}},a()}();
//# sourceMappingURL=../../../maps/libs/jquery/jstorage.js.map