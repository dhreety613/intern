async function run(endpoint){

    document.getElementById("response").innerHTML =
    "Running " + endpoint + "...";

    try{

        const response = await fetch(endpoint,{

            method:"POST"

        });

        const text = await response.text();

        document.getElementById("response").innerHTML = text;

    }

    catch(err){

        document.getElementById("response").innerHTML =
        err;

    }

}