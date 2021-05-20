/*******************************************************************************
 * Copyright (c) 2004, 2006 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.ui.tests.intro;

import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.internal.Workbench;
import org.eclipse.ui.internal.intro.IntroDescriptor;
import org.eclipse.ui.intro.IIntroPart;
import org.eclipse.ui.tests.api.IWorkbenchPartTest;
import org.eclipse.ui.tests.api.MockPart;

/**
 * @since 3.0
 */
public class NoIntroPartTest extends IWorkbenchPartTest {

    private IntroDescriptor oldDesc;

    /**
     * @param testName
     */
    public NoIntroPartTest(String testName) {
        super(testName);
        // TODO Auto-generated constructor stub
    }

    /* (non-Javadoc)
     * @see org.eclipse.ui.tests.api.IWorkbenchPartTest#openPart(org.eclipse.ui.IWorkbenchPage)
     */
    protected MockPart openPart(IWorkbenchPage page) throws Throwable {
        return (MockPart) page.getWorkbenchWindow().getWorkbench()
                .getIntroManager().showIntro(page.getWorkbenchWindow(), false);
    }

    /* (non-Javadoc)
     * @see org.eclipse.ui.tests.api.IWorkbenchPartTest#closePart(org.eclipse.ui.IWorkbenchPage, org.eclipse.ui.tests.api.MockWorkbenchPart)
     */
    protected void closePart(IWorkbenchPage page, MockPart part)
            throws Throwable {
        assertTrue(page.getWorkbenchWindow().getWorkbench().getIntroManager()
                .closeIntro((IIntroPart) part));
    }

    //only test open..shouldn't work.
    public void testOpenAndClose() throws Throwable {
        // Open a part.
        MockPart part = openPart(fPage);
        assertNull(part);
    }

    /* (non-Javadoc)
     * @see org.eclipse.ui.tests.util.UITestCase#doSetUp()
     */
    protected void doSetUp() throws Exception {
        super.doSetUp();
        oldDesc = Workbench.getInstance().getIntroDescriptor();
        Workbench.getInstance().setIntroDescriptor(null);
    }

    /* (non-Javadoc)
     * @see org.eclipse.ui.tests.util.UITestCase#doTearDown()
     */
    protected void doTearDown() throws Exception {
        super.doTearDown();
        Workbench.getInstance().setIntroDescriptor(oldDesc);
    }

}
