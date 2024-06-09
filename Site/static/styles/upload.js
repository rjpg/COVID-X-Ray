window.onload = () => {
    $('#sendbutton').click(() => {
        imagebox = $('#imagebox')
        input = $('#imageinput')[0]
        if(input.files && input.files[0])
        {
            let formData = new FormData();
            formData.append('image' , input.files[0]);
            $.ajax({
                url: "http://localhost:5000/test", // fix this to your liking
                type:"POST",
                data: formData,
                cache: false,
                processData:false,
                contentType:false,
                error: function(data){
                    console.log("upload error" , data);
                    console.log(data.getAllResponseHeaders());
                },
                success: function(data){
                    // alert("hello"); // if it's failing on actual server check your server FIREWALL + SET UP CORS
                    enviada = data['enviada']
                    recebido = data['recebido']
                    document.getElementById('enviado').src = "../BE/"+enviada;
                    document.getElementById('recebido').src = "../BE/"+recebido;
                    document.getElementById("container_img_recebida").style.display = "none";
                    document.getElementById("container_img_enviada").style.display = "";
                    $('#recebido').removeData('elevateZoom');
                    $('#enviado').elevateZoom({
                        zoomType: "inner",
                        cursor: "crosshair",
                        zoomWindowFadeIn: 500,
                        zoomWindowFadeOut: 750
                    });
                    document.getElementById("checkbox").disabled = false;
                }
            });
        }
    });
};

/*
function readUrl(input){
    imagebox = $('#imagebox')
    //console.log("evoked readUrl")
    if(input.files && input.files[0]){
        let reader = new FileReader();
        reader.onload = function(e){
            // console.log(e)
            
            imagebox.attr('src',e.target.result); 
            imagebox.height(300);
            imagebox.width(300);
        }
        reader.readAsDataURL(input.files[0]);
    }

    
}*/