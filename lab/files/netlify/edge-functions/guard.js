export default async function guard(request, context) {
  const GOOD_BOTS = [
    /Googlebot/i, /Google-InspectionTool/i, /Bingbot/i, /DuckDuckBot/i,
    /Applebot/i, /Twitterbot/i, /LinkedInBot/i, /FacebookExternalHit/i,
    /Slackbot-LinkExpanding/i
  ];

  const BAD_PATHS = [
    '/xmlrpc.php','/wp-admin/','/wordpress/','/wlwmanifest.xml','/vendor/phpunit',
    '/.env','/.git','/.DS_Store','/server-status','/wp-login.php'
  ];

  const SUS_UA = [
    /curl/i, /wget/i, /python-requests/i, /nikto/i, /sqlmap/i, /masscan/i,
    /nmap/i, /dirbuster/i, /acunetix/i, /wpscan/i
  ];

  try {
    const url = new URL(request.url);
    const ua  = request.headers.get('user-agent') || '';
    const path = url.pathname || '/';

    if (BAD_PATHS.some(p => path.toLowerCase().includes(p))) {
      return new Response('Not found', { status: 404 });
    }
    if (SUS_UA.some(r => r.test(ua))) {
      return new Response('Forbidden', { status: 403 });
    }

    const isBotUA = /\b(bot|crawler|spider|preview)\b/i.test(ua);
    const goodBot = GOOD_BOTS.some(r => r.test(ua));

    if (isBotUA && !goodBot) {
      return new Response('Robots not allowed', { status: 403 });
    }

    return context.next();
  } catch (err) {
    return new Response(`Internal error: ${err.message}`, { status: 500 });
  }
}
