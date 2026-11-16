async (image, receipt, threshold) => {
    const unchanged = [{__type__: "update"}, {__type__: "update"}];
    const origin = new URLSearchParams(location.search).get("__labs_parent");
    const allowed = ["https://app.skyebrowse.com", "https://staging.app.skyebrowse.com", "https://dev.app.skyebrowse.com", "http://localhost:8080"];
    if (window.parent === window || !allowed.includes(origin) || !receipt?.ticket || image?.path !== receipt.image) return unchanged;
    const requestId = crypto.randomUUID();
    const currentImage = document.querySelector('input[type="file"]');
    return new Promise(resolve => {
        const finish = result => {
            clearTimeout(timer);
            window.removeEventListener("message", receive);
            currentImage?.removeEventListener("change", cancel);
            document.removeEventListener("click", onClear, true);
            resolve(result);
        };
        const cancel = () => finish(unchanged);
        const onClear = event => {
            if (event.target.closest?.('button[aria-label="Remove Image"], table[role="grid"] tbody tr')) cancel();
        };
        const receive = event => {
            if (event.source !== window.parent || event.origin !== origin) return;
            const data = event.data;
            if (data?.type !== "labs:crowd-result" || data.requestId !== requestId) return;
            if (data.error) return finish([null, data.error]);
            if (!/^data:image\/(png|webp|jpeg);base64,/.test(data.overlay) || !/^[\d,]+ people$/.test(data.count)) return;
            finish([{url: data.overlay, path: data.overlay, orig_name: "crowd-count.webp", meta: {_type: "gradio.FileData"}}, data.count]);
        };
        const timer = setTimeout(() => finish([null, "Crowd Counter timed out. Please upload the image again."]), 120000);
        window.addEventListener("message", receive);
        currentImage?.addEventListener("change", cancel, {once: true});
        document.addEventListener("click", onClear, true);
        window.parent.postMessage({type: "labs:crowd-fallback", requestId, ticket: receipt.ticket, threshold}, origin);
    });
}
